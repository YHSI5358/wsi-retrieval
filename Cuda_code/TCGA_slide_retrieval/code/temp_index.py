"""
HIRES: Hierarchical Retrieval Framework for WSI Navigation

Implements Algorithm 1 (Hierarchical Index Construction) and
Algorithm 2 (GPU-Optimized Region Retrieval) from the HIRES paper.

Key components:
- Multi-resolution pyramid with vertical supporting edges
- Horizontal semantic edges via NN-Descent (K=64, cosine similarity)
- Graph pruning to uniform degree d=64
- GPU search via CUDA kernels (warp-centric + forgettable hash tables)
- Topological recomposition with area-normalized scoring
"""

import numpy as np
import torch
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import heapq
import cv2
from skimage.filters import threshold_otsu


def cosine_similarity_vectors(a, b):
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return dot / (norm_a * norm_b)


def cosine_distance_batch(query, candidates):
    """Compute cosine distance (1 - cos_sim) from query to each candidate.

    Args:
        query: [D] feature vector
        candidates: [M, D] feature matrix
    Returns:
        distances: [M] cosine distances (lower = more similar)
    """
    query_norm = np.linalg.norm(query)
    if query_norm < 1e-8:
        return np.ones(candidates.shape[0])
    cand_norms = np.linalg.norm(candidates, axis=1)
    cand_norms = np.maximum(cand_norms, 1e-8)
    dots = candidates @ query
    sims = dots / (cand_norms * query_norm)
    return 1.0 - sims


def region_similarity(features1, positions1, wh1,
                      features2, positions2, wh2):
    """Compute Region Similarity as defined in the paper (Equation 3).

    Sim(r_phi1, r_phi2) = alpha * (1/|D|) * sum cos_sim(f1_i, f2_j)

    where alpha = exp(-|w1-w2| - |h1-h2|) penalizes aspect ratio mismatch,
    and the sum is over spatially corresponding patches in the overlapping domain.

    Args:
        features1: [N1, D] features of region 1 patches
        positions1: [N1, 2] positions (in patch coordinates)
        wh1: (w, h) region 1 dimensions in patches
        features2: [N2, D] features of region 2 patches
        positions2: [N2, 2] positions (in patch coordinates)
        wh2: (w, h) region 2 dimensions in patches
    Returns:
        similarity: float, region-level similarity score
    """
    w1, h1 = wh1
    w2, h2 = wh2
    # Aspect ratio penalty
    alpha = np.exp(-abs(w1 - w2) - abs(h1 - h2))

    # Normalize positions to [0,1] domain for spatial matching
    pos1_norm = np.zeros_like(positions1, dtype=np.float64)
    pos2_norm = np.zeros_like(positions2, dtype=np.float64)
    if w1 > 0 and h1 > 0:
        pos1_min = positions1.min(axis=0)
        pos1_norm = (positions1 - pos1_min).astype(np.float64)
        pos1_norm[:, 0] /= max(w1, 1)
        pos1_norm[:, 1] /= max(h1, 1)
    if w2 > 0 and h2 > 0:
        pos2_min = positions2.min(axis=0)
        pos2_norm = (positions2 - pos2_min).astype(np.float64)
        pos2_norm[:, 0] /= max(w2, 1)
        pos2_norm[:, 1] /= max(h2, 1)

    # Match patches by nearest normalized position
    total_sim = 0.0
    n_matched = 0
    for i, p1 in enumerate(pos1_norm):
        # Find closest patch in region 2
        dists = np.linalg.norm(pos2_norm - p1, axis=1)
        j = np.argmin(dists)
        if dists[j] < 0.5:  # reasonable spatial correspondence
            sim = cosine_similarity_vectors(features1[i], features2[j])
            total_sim += sim
            n_matched += 1

    if n_matched == 0:
        return 0.0
    return alpha * (total_sim / n_matched)


class WSIVectorIndex:
    """Hierarchical Vector Index for WSI retrieval (HIRES Algorithm 1 & 2)."""

    def __init__(self, patch_size=224, feature_dim=1024, semantic_k=64, support_k=4,
                 gpu_degree=64, max_iters=20, random_seed=42, otsu_sigma=(1, 1)):
        self.patch_size = patch_size
        self.feature_dim = feature_dim
        self.semantic_k = semantic_k
        self.support_k = support_k
        self.gpu_degree = gpu_degree
        self.max_iters = max_iters
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.otsu_sigma = otsu_sigma

        self.level_graphs = {}
        self.level_info = {}
        self.global_indices = {}

    # ================================================================
    # Tissue Detection (Otsu thresholding)
    # ================================================================

    def _is_valid_patch(self, patch_image):
        if len(patch_image.shape) == 3:
            patch_gray = cv2.cvtColor(patch_image, cv2.COLOR_RGB2GRAY)
        else:
            patch_gray = patch_image
        patch_blur = cv2.GaussianBlur(patch_gray, self.otsu_sigma, 0)
        try:
            thresh = threshold_otsu(patch_blur)
            binary = patch_blur > thresh
            tissue_ratio = np.mean(binary)
            return tissue_ratio > 0.1
        except:
            return False

    # ================================================================
    # Algorithm 1: Hierarchical Index Construction
    # ================================================================

    def add_level(self, features, positions, level, wsi_dimensions, patch_images=None):
        """Add a magnification level to the index."""
        assert features.shape[1] == self.feature_dim
        assert len(positions) == features.shape[0]

        if patch_images is not None:
            valid_indices = [i for i, img in enumerate(patch_images)
                            if self._is_valid_patch(img)]
            features = features[valid_indices]
            positions = [positions[i] for i in valid_indices]

        self.level_graphs[level] = {
            'features': features,
            'positions': np.array(positions),
            'graph': None
        }
        self.level_info[level] = wsi_dimensions

    def _build_supporting_connections(self):
        """Build vertical supporting edges between magnification levels."""
        supporting_conn = {}
        levels = sorted(self.level_graphs.keys())

        for i, level in enumerate(levels):
            current_pos = self.level_graphs[level]['positions']

            for node_idx, pos in enumerate(current_pos):
                supporting_conn[(level, node_idx)] = []

                # Upward connections (to lower-magnification parent)
                if i > 0:
                    higher_level = levels[i - 1]
                    higher_pos = self.level_graphs[higher_level]['positions']
                    scale_factor = 2 ** (higher_level - level)
                    x, y = pos
                    higher_x_min = x * scale_factor
                    higher_y_min = y * scale_factor
                    higher_x_max = (x + 1) * scale_factor
                    higher_y_max = (y + 1) * scale_factor

                    for higher_idx, higher_p in enumerate(higher_pos):
                        hx, hy = higher_p
                        if (higher_x_min <= hx < higher_x_max and
                                higher_y_min <= hy < higher_y_max):
                            supporting_conn[(level, node_idx)].append((higher_level, higher_idx))
                            if len(supporting_conn[(level, node_idx)]) >= self.support_k:
                                break

                # Downward connections (to higher-magnification children)
                if i < len(levels) - 1:
                    lower_level = levels[i + 1]
                    lower_pos = self.level_graphs[lower_level]['positions']
                    scale_factor = 2 ** (level - lower_level)
                    x, y = pos
                    lower_x = x // scale_factor
                    lower_y = y // scale_factor

                    for lower_idx, lower_p in enumerate(lower_pos):
                        lx, ly = lower_p
                        if lx == lower_x and ly == lower_y:
                            supporting_conn[(level, node_idx)].append((lower_level, lower_idx))
                            break

        return supporting_conn

    def _build_semantic_connections(self, use_nndescent=True):
        """Build horizontal semantic edges using K-NN with cosine similarity.

        FIX: Changed metric from 'euclidean' to 'cosine' to match paper.
        """
        semantic_conn = {}

        for level in self.level_graphs:
            features = self.level_graphs[level]['features']
            n_samples = features.shape[0]

            if use_nndescent:
                from pynndescent import NNDescent
                nnd = NNDescent(features,
                                n_neighbors=self.semantic_k,
                                metric='cosine',       # FIX: was 'euclidean'
                                n_jobs=-1,
                                random_state=self.random_seed)
                indices, distances = nnd.neighbor_graph
            else:
                knn = NearestNeighbors(n_neighbors=self.semantic_k,
                                       algorithm='auto',
                                       metric='cosine',  # FIX: was 'euclidean'
                                       n_jobs=-1)
                knn.fit(features)
                distances, indices = knn.kneighbors(features)

            for i in range(n_samples):
                semantic_conn[(level, i)] = [(level, idx) for idx in indices[i]]

        return semantic_conn

    def _prune_graph(self, supporting_conn, semantic_conn):
        """Prune graph to uniform degree d=64.

        Priority 1: Keep ALL vertical supporting edges.
        Priority 2: Fill remaining slots with top semantic edges by cosine similarity.
        FIX: Pad under-degree nodes to exactly gpu_degree with random same-level nodes.
        FIX: Use cosine similarity for ranking (not euclidean distance).
        """
        uniform_graph = {}

        # Build global index mapping
        global_idx = 0
        self.global_indices = {}
        for level in sorted(self.level_graphs.keys()):
            n_nodes = len(self.level_graphs[level]['features'])
            self.global_indices[level] = (global_idx, global_idx + n_nodes)
            global_idx += n_nodes

        for level in self.level_graphs:
            n_nodes = len(self.level_graphs[level]['features'])

            for node_idx in range(n_nodes):
                key = (level, node_idx)

                all_connections = supporting_conn.get(key, []) + semantic_conn.get(key, [])
                unique_connections = list(set([conn for conn in all_connections if conn != key]))

                if len(unique_connections) > self.gpu_degree:
                    # Need to prune: rank by cosine similarity
                    current_feat = self.level_graphs[level]['features'][node_idx]
                    conn_feats = np.array([
                        self.level_graphs[cl]['features'][ci]
                        for (cl, ci) in unique_connections
                    ])
                    # FIX: Use cosine similarity (higher = better, so sort descending)
                    similarities = conn_feats @ current_feat / (
                        np.linalg.norm(conn_feats, axis=1) * np.linalg.norm(current_feat) + 1e-8
                    )
                    sorted_indices = np.argsort(-similarities)  # descending

                    supporting_set = set(supporting_conn.get(key, []))
                    kept_connections = []

                    # Priority 1: all supporting edges
                    for conn in unique_connections:
                        if conn in supporting_set:
                            kept_connections.append(conn)

                    # Priority 2: top semantic edges by cosine similarity
                    for idx in sorted_indices:
                        conn = unique_connections[idx]
                        if conn not in supporting_set and len(kept_connections) < self.gpu_degree:
                            kept_connections.append(conn)

                    uniform_graph[key] = kept_connections[:self.gpu_degree]

                elif len(unique_connections) < self.gpu_degree:
                    # FIX: Pad to uniform degree with random same-level nodes
                    kept = list(unique_connections)
                    existing_set = set(unique_connections) | {key}
                    deficit = self.gpu_degree - len(kept)
                    # Sample random nodes from same level
                    same_level_nodes = [(level, i) for i in range(n_nodes)
                                        if (level, i) not in existing_set]
                    if same_level_nodes and deficit > 0:
                        n_pad = min(deficit, len(same_level_nodes))
                        pad_indices = np.random.choice(len(same_level_nodes), n_pad, replace=False)
                        for pi in pad_indices:
                            kept.append(same_level_nodes[pi])
                    uniform_graph[key] = kept
                else:
                    uniform_graph[key] = unique_connections

        return uniform_graph

    def build_index(self):
        """Build the complete hierarchical index (Algorithm 1)."""
        supporting_conn = self._build_supporting_connections()
        semantic_conn = self._build_semantic_connections()
        self.graph = self._prune_graph(supporting_conn, semantic_conn)

        self.global_to_level = {}
        for level, (start, end) in self.global_indices.items():
            for i in range(end - start):
                self.global_to_level[start + i] = (level, i)

        return self

    # ================================================================
    # Algorithm 2: GPU-Optimized Region Retrieval
    # ================================================================

    def _prepare_gpu_data(self):
        """Prepare flattened features and adjacency matrix for GPU."""
        if not hasattr(self, '_gpu_features_np'):
            all_features = []
            for level in sorted(self.level_graphs.keys()):
                all_features.append(self.level_graphs[level]['features'])
            self._gpu_features_np = np.vstack(all_features).astype(np.float32)

            n_nodes = self._gpu_features_np.shape[0]
            self._gpu_adjacency_np = np.full((n_nodes, self.gpu_degree), -1, dtype=np.int32)
            for (level, idx), neighbors in self.graph.items():
                global_idx = self.global_indices[level][0] + idx
                for i, (neighbor_level, neighbor_idx) in enumerate(neighbors):
                    if i < self.gpu_degree:
                        neighbor_global_idx = self.global_indices[neighbor_level][0] + neighbor_idx
                        self._gpu_adjacency_np[global_idx, i] = neighbor_global_idx

    def _get_gpu_searcher(self):
        """Lazily create the CUDA kernel-based GPU searcher."""
        if not hasattr(self, '_gpu_searcher'):
            self._prepare_gpu_data()
            from cuda_kernels import HiresGPUSearcher
            self._gpu_searcher = HiresGPUSearcher(
                features_np=self._gpu_features_np,
                adjacency_np=self._gpu_adjacency_np,
                degree=self.gpu_degree,
                feature_dim=self.feature_dim,
                hash_table_size=512,
                hash_reset_interval=4,
                max_iters=self.max_iters,
                exploration_p=2,
                top_m=200
            )
        return self._gpu_searcher

    def search_gpu(self, query_feature, top_m=200, exploration_factor=2):
        """GPU-accelerated search using CUDA kernels with warp-centric computation.

        FIX: Now uses actual CUDA kernels instead of PyTorch simulation.
        FIX: Default top_m=200 to match paper.
        FIX: Uses cosine similarity (higher = better).

        Args:
            query_feature: [D] query vector
            top_m: number of nearest neighbors to return (paper default: 200)
            exploration_factor: number of seed nodes per iteration
        Returns:
            list of (level, idx, similarity) tuples, sorted by similarity descending
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        if not hasattr(self, 'graph') or not hasattr(self, 'global_to_level'):
            raise ValueError("Index not built yet. Call build_index() first.")

        searcher = self._get_gpu_searcher()
        query_np = np.array(query_feature, dtype=np.float32).reshape(1, -1)

        sims, idxs = searcher.search(query_np, top_m=top_m)

        # Convert global indices back to (level, idx, similarity)
        results = []
        for i in range(top_m):
            node_idx = int(idxs[0, i])
            sim = float(sims[0, i])
            if node_idx >= 0 and node_idx in self.global_to_level:
                level, idx = self.global_to_level[node_idx]
                results.append((level, idx, sim))

        # Sort by similarity descending (higher = better)
        results.sort(key=lambda x: -x[2])
        return results

    def search_region_gpu(self, query_features, positions, top_m=200, exploration_factor=2):
        """Batch GPU search for all patches in a query region.

        FIX: Now truly parallel - all query patches searched in one kernel launch.
        FIX: Default top_m=200 to match paper.

        Args:
            query_features: [N, D] features of all patches in query region
            positions: [N, 2] positions of patches
            top_m: neighbors per patch
            exploration_factor: exploration parameter
        Returns:
            dict mapping position tuple -> list of (level, idx, similarity)
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        if not hasattr(self, 'graph') or not hasattr(self, 'global_to_level'):
            raise ValueError("Index not built yet. Call build_index() first.")

        searcher = self._get_gpu_searcher()
        query_batch = np.array(query_features, dtype=np.float32)

        # Single kernel launch for ALL query patches (truly parallel)
        all_sims, all_idxs = searcher.search(query_batch, top_m=top_m)

        results = {}
        for i, pos in enumerate(positions):
            patch_results = []
            for j in range(top_m):
                node_idx = int(all_idxs[i, j])
                sim = float(all_sims[i, j])
                if node_idx >= 0 and node_idx in self.global_to_level:
                    level, idx = self.global_to_level[node_idx]
                    patch_results.append((level, idx, sim))
            patch_results.sort(key=lambda x: -x[2])
            results[tuple(pos)] = patch_results

        return results

    # ================================================================
    # Stage 3: Topological Recomposition
    # ================================================================

    def aggregate_region_results(self, region_results, query_features=None):
        """Topological recomposition with area-normalized scoring.

        FIX: Score_j = sum(Sim(p_i, v_q)) / Area(B_j) per paper Algorithm 2.
        Was: confidence = 1/(1+mean_dist)

        Args:
            region_results: dict from search_region_gpu, pos -> [(level, idx, sim), ...]
            query_features: optional [N, D] query features for re-scoring
        Returns:
            list of dicts with 'level', 'bbox', 'score', sorted by score descending
        """
        # Group candidate patches by (wsi_source, level)
        # Each result patch is identified by (level, idx)
        # Group by level first, then find connected components
        level_results = {}
        for pos, results in region_results.items():
            for level, idx, sim in results:
                if level not in level_results:
                    level_results[level] = []
                level_results[level].append({
                    'query_pos': pos,
                    'result_idx': idx,
                    'similarity': sim,
                    'result_pos': tuple(self.level_graphs[level]['positions'][idx])
                })

        bounding_boxes = []
        for level, results in level_results.items():
            if not results:
                continue

            # Find connected components by spatial adjacency
            result_positions = [r['result_pos'] for r in results]
            components = self._find_connected_components(result_positions)

            for component_indices in components:
                comp_results = [results[i] for i in component_indices]
                positions = np.array([r['result_pos'] for r in comp_results])
                similarities = [r['similarity'] for r in comp_results]

                min_x, min_y = positions.min(axis=0)
                max_x, max_y = positions.max(axis=0)

                # Area in patch units (paper: Area(B_j))
                area = (max_x - min_x + 1) * (max_y - min_y + 1)

                # Score = sum(Sim) / Area(B_j)  (paper Eq. in Algorithm 2)
                score = sum(similarities) / area if area > 0 else 0.0

                bounding_boxes.append({
                    'level': level,
                    'bbox': (int(min_x), int(min_y), int(max_x), int(max_y)),
                    'score': score,
                    'n_patches': len(comp_results)
                })

        # Sort by score descending (higher = better)
        bounding_boxes.sort(key=lambda x: -x['score'])
        return bounding_boxes

    def _find_connected_components(self, positions):
        """Find connected components among positions via 8-directional adjacency."""
        n = len(positions)
        if n == 0:
            return []

        pos_set = {}
        for i, p in enumerate(positions):
            pos_set.setdefault(p, []).append(i)

        visited = [False] * n
        components = []

        for i in range(n):
            if visited[i]:
                continue
            component = []
            stack = [i]
            while stack:
                node = stack.pop()
                if visited[node]:
                    continue
                visited[node] = True
                component.append(node)
                px, py = positions[node]
                # 8-directional neighbors
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        neighbor_pos = (px + dx, py + dy)
                        if neighbor_pos in pos_set:
                            for ni in pos_set[neighbor_pos]:
                                if not visited[ni]:
                                    stack.append(ni)
            if len(component) > 0:
                components.append(component)

        return components

    # ================================================================
    # CPU Search (reference implementation)
    # ================================================================

    def search(self, query_feature, top_k=200, exploration_factor=2, use_gpu=False):
        """Search for nearest neighbors.

        FIX: Default top_k=200 to match paper's m=200.
        FIX: Uses cosine similarity instead of euclidean distance.
        """
        if use_gpu and torch.cuda.is_available():
            return self.search_gpu(query_feature, top_m=top_k,
                                   exploration_factor=exploration_factor)

        if not hasattr(self, 'graph') or not hasattr(self, 'global_to_level'):
            raise ValueError("Index not built yet. Call build_index() first.")

        # Build global graph
        global_graph = {}
        for (level, idx), neighbors in self.graph.items():
            gi = self.global_indices[level][0] + idx
            global_graph[gi] = [
                self.global_indices[nl][0] + ni for (nl, ni) in neighbors
            ]

        # Get flattened features for fast lookup
        self._prepare_gpu_data()
        all_features = self._gpu_features_np

        p = exploration_factor
        all_nodes = list(global_graph.keys())
        n_init = min(p * self.gpu_degree, len(all_nodes))
        init_nodes = np.random.choice(all_nodes, size=n_init, replace=False)

        # Use cosine similarity (higher = better) stored as negative for min-heap
        candidate_list = []
        visited_set = set()
        for node in init_nodes:
            sim = cosine_similarity_vectors(query_feature, all_features[node])
            heapq.heappush(candidate_list, (-sim, node))  # negative for max-heap via min-heap
            visited_set.add(node)

        result_list = []  # [(neg_sim, node)]
        result_updated = True
        iter_count = 0

        while result_updated and iter_count < self.max_iters:
            iter_count += 1
            result_updated = False

            new_result_list = []
            temp_candidates = list(candidate_list)
            temp_candidates.sort()
            for neg_sim, node in temp_candidates[:top_k]:
                new_result_list.append((neg_sim, node))

            if not result_list or new_result_list[0][0] < result_list[0][0]:
                result_updated = True
                result_list = new_result_list

            if not result_updated:
                break

            top_p_nodes = result_list[:p]

            new_candidates = []
            for neg_sim, node in top_p_nodes:
                for neighbor in global_graph.get(node, []):
                    if neighbor not in visited_set:
                        sim = cosine_similarity_vectors(query_feature, all_features[neighbor])
                        new_candidates.append((-sim, neighbor))
                        visited_set.add(neighbor)

            candidate_list = sorted(new_candidates)[:p * self.gpu_degree]

        formatted_results = []
        for neg_sim, node in result_list:
            level, idx = self.global_to_level[node]
            formatted_results.append((level, idx, -neg_sim))  # return positive similarity

        # Sort by similarity descending
        formatted_results.sort(key=lambda x: -x[2])
        return formatted_results

    def search_region(self, query_features, positions, top_k=200,
                      exploration_factor=2, use_gpu=False):
        """Search for a query region (multiple patches).

        FIX: Default top_k=200.
        """
        if use_gpu and torch.cuda.is_available():
            return self.search_region_gpu(query_features, positions,
                                          top_m=top_k, exploration_factor=exploration_factor)

        results = {}
        for feature, pos in zip(query_features, positions):
            patch_results = self.search(feature, top_k=top_k,
                                        exploration_factor=exploration_factor)
            results[tuple(pos)] = patch_results

        return results

    # ================================================================
    # Serialization
    # ================================================================

    def save_index(self, filepath):
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'patch_size': self.patch_size,
                'feature_dim': self.feature_dim,
                'semantic_k': self.semantic_k,
                'support_k': self.support_k,
                'gpu_degree': self.gpu_degree,
                'level_graphs': self.level_graphs,
                'level_info': self.level_info,
                'graph': self.graph,
                'global_indices': self.global_indices,
                'global_to_level': self.global_to_level
            }, f)

    @classmethod
    def load_index(cls, filepath):
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        index = cls(
            patch_size=data['patch_size'],
            feature_dim=data['feature_dim'],
            semantic_k=data['semantic_k'],
            support_k=data['support_k'],
            gpu_degree=data['gpu_degree']
        )

        index.level_graphs = data['level_graphs']
        index.level_info = data['level_info']
        index.graph = data['graph']
        index.global_indices = data['global_indices']
        index.global_to_level = data['global_to_level']

        return index

    # ================================================================
    # Testing / Benchmarking
    # ================================================================

    def test_otsu(self, patch_images):
        results = []
        for img in patch_images:
            is_valid = self._is_valid_patch(img)
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img
            blur = cv2.GaussianBlur(gray, self.otsu_sigma, 0)
            try:
                thresh = threshold_otsu(blur)
                binary = blur > thresh
                tissue_ratio = np.mean(binary)
            except:
                tissue_ratio = 0.0
            results.append((is_valid, tissue_ratio))
        return results

    def test_graph_construction(self):
        """Verify graph has uniform degree."""
        degrees = []
        for key, neighbors in self.graph.items():
            degrees.append(len(neighbors))
        degrees = np.array(degrees)
        return {
            'n_nodes': len(self.graph),
            'avg_degree': np.mean(degrees),
            'max_degree': np.max(degrees),
            'min_degree': np.min(degrees),
            'target_degree': self.gpu_degree,
            'uniform': np.all(degrees == self.gpu_degree)
        }

    def test_search_accuracy(self, query_features, ground_truth, top_k=10, use_gpu=False):
        correct = 0
        retrieved = 0
        relevant = len(ground_truth)

        for query in query_features:
            results = self.search(query, top_k=top_k, use_gpu=use_gpu)
            retrieved_ids = {(level, idx) for level, idx, _ in results}
            correct += len(retrieved_ids & set(ground_truth))
            retrieved += len(retrieved_ids)

        precision = correct / retrieved if retrieved > 0 else 0
        recall = correct / relevant if relevant > 0 else 0
        return {'precision@k': precision, 'recall@k': recall}

    def benchmark_gpu_vs_cpu(self, query_features, top_k=10, exploration_factor=2):
        import time

        if torch.cuda.is_available():
            self.search(query_features[0], top_k=top_k,
                        exploration_factor=exploration_factor, use_gpu=True)

        cpu_times = []
        for query in query_features:
            start = time.time()
            self.search(query, top_k=top_k,
                        exploration_factor=exploration_factor, use_gpu=False)
            cpu_times.append(time.time() - start)

        gpu_times = []
        if torch.cuda.is_available():
            for query in query_features:
                start = time.time()
                self.search(query, top_k=top_k,
                            exploration_factor=exploration_factor, use_gpu=True)
                gpu_times.append(time.time() - start)

        return {
            'cpu_avg_time': np.mean(cpu_times),
            'gpu_avg_time': np.mean(gpu_times) if gpu_times else None,
            'speedup': np.mean(cpu_times) / np.mean(gpu_times) if gpu_times else None
        }

    def save_test_data(self, filepath, sample_size=1000):
        import pickle
        test_data = {
            'params': {
                'patch_size': self.patch_size,
                'feature_dim': self.feature_dim,
                'semantic_k': self.semantic_k,
                'support_k': self.support_k,
                'gpu_degree': self.gpu_degree
            },
            'sample_features': {},
            'sample_graph': {}
        }
        for level in self.level_graphs:
            features = self.level_graphs[level]['features']
            n_samples = min(sample_size, features.shape[0])
            idx = np.random.choice(features.shape[0], n_samples, replace=False)
            test_data['sample_features'][level] = features[idx]
            sampled_graph = {}
            for i in idx:
                key = (level, i)
                if key in self.graph:
                    sampled_graph[key] = self.graph[key]
            test_data['sample_graph'][level] = sampled_graph
        with open(filepath, 'wb') as f:
            pickle.dump(test_data, f)
