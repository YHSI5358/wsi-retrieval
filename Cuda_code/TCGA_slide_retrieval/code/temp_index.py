import numpy as np
import torch
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import heapq
import hashlib
import cv2
from skimage.filters import threshold_otsu

class WSIVectorIndex:
    def __init__(self, patch_size=224, feature_dim=1024, semantic_k=64, support_k=4,
                 gpu_degree=64, max_iters=20, random_seed=42, otsu_sigma=(1,1)):
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

    def add_level(self, features, positions, level, wsi_dimensions, patch_images=None):

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

        supporting_conn = {}
        levels = sorted(self.level_graphs.keys())
        
        for i, level in enumerate(levels):
            current_graph = self.level_graphs[level]
            current_pos = current_graph['positions']
            
            
            for node_idx, pos in enumerate(current_pos):
                supporting_conn[(level, node_idx)] = []
                
                
                if i > 0:
                    higher_level = levels[i-1]
                    higher_graph = self.level_graphs[higher_level]
                    higher_pos = higher_graph['positions']
                    
                    
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
                
                
                if i < len(levels) - 1:
                    lower_level = levels[i+1]
                    lower_graph = self.level_graphs[lower_level]
                    lower_pos = lower_graph['positions']
                    
                    
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

        semantic_conn = {}
        
        for level in self.level_graphs:
            features = self.level_graphs[level]['features']
            n_samples = features.shape[0]
            
            
            if use_nndescent:
                
                from pynndescent import NNDescent
                nnd = NNDescent(features, 
                               n_neighbors=self.semantic_k,
                               metric='euclidean',
                               n_jobs=-1,
                               random_state=self.random_seed)
                indices, distances = nnd.neighbor_graph
            else:
                
                knn = NearestNeighbors(n_neighbors=self.semantic_k, 
                                     algorithm='auto',
                                     metric='euclidean',
                                     n_jobs=-1)
                knn.fit(features)
                distances, indices = knn.kneighbors(features)
            
            
            for i in range(n_samples):
                semantic_conn[(level, i)] = [(level, idx) for idx in indices[i]]
        
        return semantic_conn
    
    def _prune_graph(self, supporting_conn, semantic_conn):

        uniform_graph = {}
        
        
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
                    
                    current_feat = self.level_graphs[level]['features'][node_idx]
                    conn_feats = []
                    for (conn_level, conn_idx) in unique_connections:
                        conn_feats.append(self.level_graphs[conn_level]['features'][conn_idx])
                    conn_feats = np.array(conn_feats)
                    
                    
                    distances = np.linalg.norm(conn_feats - current_feat, axis=1)
                    sorted_indices = np.argsort(distances)
                    
                    
                    supporting_set = set(supporting_conn.get(key, []))
                    kept_connections = []
                    
                    
                    for conn in unique_connections:
                        if conn in supporting_set:
                            kept_connections.append(conn)
                    
                    
                    for idx in sorted_indices:
                        conn = unique_connections[idx]
                        if conn not in supporting_set and len(kept_connections) < self.gpu_degree:
                            kept_connections.append(conn)
                    
                    uniform_graph[key] = kept_connections[:self.gpu_degree]
                else:
                    uniform_graph[key] = unique_connections
        
        return uniform_graph
    
    def build_index(self):

        supporting_conn = self._build_supporting_connections()
        
        semantic_conn = self._build_semantic_connections()
        
        self.graph = self._prune_graph(supporting_conn, semantic_conn)
        
        
        self.global_to_level = {}
        for level, (start, end) in self.global_indices.items():
            for i in range(end - start):
                self.global_to_level[start + i] = (level, i)
        
        return self
    
    def _ensure_cuda_available(self):

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA acceleration requested but CUDA is not available on this system")
    
    def search_gpu(self, query_feature, top_k=10, exploration_factor=2, 
                  use_warp_splitting=True, hash_table_size=512):

        self._ensure_cuda_available()
        
        if not hasattr(self, 'graph') or not hasattr(self, 'global_to_level'):
            raise ValueError("Index not built yet. Call build_index() first.")
            
        
        query_tensor = torch.tensor(query_feature, dtype=torch.float32).cuda()
        
        
        if not hasattr(self, 'features_tensor_gpu'):
            
            all_features = []
            for level in sorted(self.level_graphs.keys()):
                all_features.append(self.level_graphs[level]['features'])
            all_features = np.vstack(all_features)
            self.features_tensor_gpu = torch.tensor(all_features, dtype=torch.float32).cuda()
        
        
        if not hasattr(self, 'graph_gpu'):
            
            n_nodes = self.features_tensor_gpu.shape[0]
            
            adjacency = np.full((n_nodes, self.gpu_degree), -1, dtype=np.int32)
            
            
            for (level, idx), neighbors in self.graph.items():
                global_idx = self.global_indices[level][0] + idx
                for i, (neighbor_level, neighbor_idx) in enumerate(neighbors):
                    if i < self.gpu_degree:  
                        neighbor_global_idx = self.global_indices[neighbor_level][0] + neighbor_idx
                        adjacency[global_idx, i] = neighbor_global_idx
            
            self.graph_gpu = torch.tensor(adjacency, dtype=torch.int32).cuda()
        
        
        p = exploration_factor
        if use_warp_splitting:
            
            threads_per_block = 256  
            teams_per_warp = 4
            threads_per_team = 8
        else:
            threads_per_block = 256
        blocks = (p * self.gpu_degree + threads_per_block - 1) // threads_per_block
        
        
        n_nodes = self.features_tensor_gpu.shape[0]
        
        
        top_k_results = torch.full((top_k, 2), float('inf'), dtype=torch.float32, device='cuda')
        if hash_table_size > 0:
            
            hash_tables = torch.zeros((blocks, hash_table_size), dtype=torch.long, device='cuda')
            hash_table_usage = torch.zeros(blocks, dtype=torch.int, device='cuda')
        else:
            visited = torch.zeros(n_nodes, dtype=torch.bool, device='cuda')
        
        
        init_nodes = torch.randint(0, n_nodes, (p * self.gpu_degree,), device='cuda')
        candidate_list = torch.zeros((p * self.gpu_degree, 2), dtype=torch.float32, device='cuda')  
        
        
        
        
        
        
        for i in range(init_nodes.size(0)):
            node_idx = init_nodes[i].item()
            visited[node_idx] = True
            
            
            node_feature = self.features_tensor_gpu[node_idx]
            dist = torch.norm(query_tensor - node_feature).item()
            
            candidate_list[i, 0] = dist
            candidate_list[i, 1] = node_idx
        
        
        candidate_list = candidate_list[candidate_list[:, 0].argsort()]
        
        result_updated = True
        max_iters = 20  
        iter_count = 0
        
        while result_updated and iter_count < max_iters:
            iter_count += 1
            result_updated = False
            
            
            new_results = torch.cat([top_k_results, candidate_list[:top_k]], dim=0)
            new_results = new_results[new_results[:, 0].argsort()][:top_k]
            
            
            if not torch.allclose(top_k_results, new_results):
                result_updated = True
                top_k_results = new_results
            
            if not result_updated:
                break
                
            
            top_p = min(p, top_k_results.size(0))
            
            
            candidate_list = torch.full((p * self.gpu_degree, 2), float('inf'), dtype=torch.float32, device='cuda')
            candidates_added = 0
            
            
            
            if iter_count % 4 == 0:
                
                visited_copy = visited.clone()
                visited.fill_(False)
                for i in range(top_k_results.size(0)):
                    if top_k_results[i, 0] != float('inf'):
                        node_idx = int(top_k_results[i, 1].item())
                        visited[node_idx] = True
            
            for i in range(top_p):
                node_idx = int(top_k_results[i, 1].item())
                
                
                neighbors = self.graph_gpu[node_idx]
                
                
                for j in range(self.gpu_degree):
                    neighbor_idx = neighbors[j].item()
                    if neighbor_idx == -1:  
                        continue
                        
                    if not visited[neighbor_idx]:
                        visited[neighbor_idx] = True
                        
                        
                        neighbor_feature = self.features_tensor_gpu[neighbor_idx]
                        dist = torch.norm(query_tensor - neighbor_feature).item()
                        
                        if candidates_added < candidate_list.size(0):
                            candidate_list[candidates_added, 0] = dist
                            candidate_list[candidates_added, 1] = neighbor_idx
                            candidates_added += 1
            
            
            if candidates_added > 0:
                candidate_list = candidate_list[candidate_list[:, 0].argsort()]
        
        
        results = []
        for i in range(top_k):
            if top_k_results[i, 0] != float('inf'):
                dist = top_k_results[i, 0].item()
                node_idx = int(top_k_results[i, 1].item())
                level, idx = self.global_to_level[node_idx]
                results.append((level, idx, dist))
        
        
        results.sort(key=lambda x: x[2])
        
        return results
    
    def search_region_gpu(self, query_features, positions, top_k=10, exploration_factor=2):

        self._ensure_cuda_available()
        
        
        query_batch = torch.tensor(np.array(query_features), dtype=torch.float32).cuda()
        
        results = {}
        batch_size = 32  
        
        for i in range(0, len(positions), batch_size):
            batch_positions = positions[i:i+batch_size]
            batch_queries = query_batch[i:i+batch_size]
            
            
            
            for j, (query, pos) in enumerate(zip(batch_queries, batch_positions)):
                patch_results = self.search_gpu(query.cpu().numpy(), top_k=top_k, exploration_factor=exploration_factor)
                results[tuple(pos)] = patch_results
                
        return results
    
    def aggregate_region_results(self, region_results, threshold=0.95):
        
        
        
        
        level_results = {}
        for pos, results in region_results.items():
            for level, idx, dist in results:
                if level not in level_results:
                    level_results[level] = []
                level_results[level].append((pos, idx, dist))
        
        
        bounding_boxes = []
        for level, results in level_results.items():
            if not results:
                continue
                
            
            positions = np.array([pos for pos, _, _ in results])
            min_x = np.min(positions[:, 0])
            min_y = np.min(positions[:, 1])
            max_x = np.max(positions[:, 0])
            max_y = np.max(positions[:, 1])
            
            
            confidence = 1.0 / (1.0 + np.mean([dist for _, _, dist in results]))
            
            bounding_boxes.append({
                'level': level,
                'bbox': (min_x, min_y, max_x, max_y),
                'confidence': confidence
            })
        
        return bounding_boxes
    
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

    def test_graph_construction(self, level=0):

        if level not in self.level_graphs:
            raise ValueError(f"Level {level} not found in index")
            
        graph = self.level_graphs[level]['graph']
        n_nodes = len(self.level_graphs[level]['features'])
        
        
        degrees = [len(neighbors) for neighbors in graph.values()]
        
        return {
            'n_nodes': n_nodes,
            'avg_degree': np.mean(degrees),
            'max_degree': np.max(degrees),
            'min_degree': np.min(degrees),
            'support_connections': self.support_k,
            'semantic_connections': self.semantic_k
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

    def search(self, query_feature, top_k=10, exploration_factor=2, use_gpu=False):

        if use_gpu and torch.cuda.is_available():
            return self.search_gpu(query_feature, top_k, exploration_factor)
            
        if not hasattr(self, 'graph') or not hasattr(self, 'global_to_level'):
            raise ValueError("Index not built yet. Call build_index() first.")
            
        
        result_list = []  
        visited_set = set()  
        candidate_list = []  
        
        
        global_graph = {}
        for (level, idx), neighbors in self.graph.items():
            global_idx = self.global_indices[level][0] + idx
            global_neighbors = []
            for (n_level, n_idx) in neighbors:
                global_neighbors.append(self.global_indices[n_level][0] + n_idx)
            global_graph[global_idx] = global_neighbors
        
        
        p = exploration_factor  
        all_nodes = list(global_graph.keys())
        init_nodes = np.random.choice(all_nodes, size=p*self.gpu_degree, replace=False)
        
        
        for node in init_nodes:
            level, idx = self.global_to_level[node]
            dist = np.linalg.norm(query_feature - self.level_graphs[level]['features'][idx])
            heapq.heappush(candidate_list, (dist, node))
            visited_set.add(node)
            
        result_updated = True
        
        
        while result_updated:
            result_updated = False
            
            
            new_result_list = []
            while candidate_list and len(new_result_list) < top_k:
                dist, node = heapq.heappop(candidate_list)
                new_result_list.append((dist, node))
                
            
            if not result_list or any(new[0] < old[0] for new, old in zip(new_result_list, result_list)):
                result_updated = True
                result_list = new_result_list
            
            if not result_updated:
                break
                
            
            top_p_nodes = result_list[:p]
            
            
            new_candidates = []
            for _, node in top_p_nodes:
                for neighbor in global_graph.get(node, []):
                    if neighbor not in visited_set:
                        level, idx = self.global_to_level[neighbor]
                        dist = np.linalg.norm(query_feature - self.level_graphs[level]['features'][idx])
                        new_candidates.append((dist, neighbor))
                        visited_set.add(neighbor)
            
            
            candidate_list = sorted(new_candidates, key=lambda x: x[0])[:p*self.gpu_degree]
            
        
        formatted_results = []
        for dist, node in result_list:
            level, idx = self.global_to_level[node]
            formatted_results.append((level, idx, dist))
            
        return formatted_results

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
            'speedup': np.mean(cpu_times)/np.mean(gpu_times) if gpu_times else None
        }

    def search_region(self, query_features, positions, top_k=10, exploration_factor=2, use_gpu=False):

        if use_gpu and torch.cuda.is_available():
            return self.search_region_gpu(query_features, positions, top_k, exploration_factor)
            
        results = {}
        
        
        for i, (feature, pos) in enumerate(zip(query_features, positions)):
            patch_results = self.search(feature, top_k=top_k, exploration_factor=exploration_factor)
            results[tuple(pos)] = patch_results
            
        return results
        
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

    
    def _cuda_kernel_simulation(self, query_batch, top_k=10, exploration_factor=2):

        
        results = []
        p = exploration_factor
        
        for query_idx in range(query_batch.size(0)):
            query = query_batch[query_idx]
            
            
            
            
            
            
            n_nodes = self.features_tensor_gpu.shape[0]
            n_warps = 4  
            threads_per_warp = 32
            
            
            visited = torch.zeros(n_nodes, dtype=torch.bool, device='cuda')
            
            
            init_nodes = torch.randint(0, n_nodes, (p * self.gpu_degree,), device='cuda')
            candidate_list = torch.zeros((p * self.gpu_degree, 2), dtype=torch.float32, device='cuda')
            
            
            for i in range(init_nodes.size(0)):
                node_idx = init_nodes[i].item()
                visited[node_idx] = True
                
                
                
                node_feature = self.features_tensor_gpu[node_idx]
                dist = torch.norm(query - node_feature).item()
                
                candidate_list[i, 0] = dist
                candidate_list[i, 1] = node_idx
            
            
            candidate_list = candidate_list[candidate_list[:, 0].argsort()]
            
            
            
            
            
            query_results = []
            for i in range(min(top_k, candidate_list.size(0))):
                if candidate_list[i, 0] != float('inf'):
                    dist = candidate_list[i, 0].item()
                    node_idx = int(candidate_list[i, 1].item())
                    level, idx = self.global_to_level[node_idx]
                    query_results.append((level, idx, dist))
            
            results.append(query_results)
        
        return results
