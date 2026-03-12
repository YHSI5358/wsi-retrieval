"""
HIRES Region-Level Evaluation Script

Implements the region-level evaluation protocol from the paper (Section 6.2):
- Test region sizes: 10x10, 20x10, 20x20 patches
- 10 random queries per size
- 10 random initializations per query
- Reports Sim@k averaged across all trials
- Compares with brute-force baseline (timing + accuracy)
"""

import os
import sys
import json
import time
import pickle
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))
from temp_index import WSIVectorIndex, region_similarity, cosine_similarity_vectors
from eval import (calculate_sim_at_k, brute_force_search,
                  brute_force_batch_search, calculate_acc_at_k, calculate_mvacc5)


def find_valid_query_regions(positions, region_w, region_h, n_queries=10):
    """Find contiguous blocks of tissue patches at the specified size.

    Args:
        positions: [N, 2] array of patch positions
        region_w: region width in patches
        region_h: region height in patches
        n_queries: number of regions to sample
    Returns:
        list of (start_x, start_y) for valid query regions
    """
    pos_set = set(map(tuple, positions))

    max_x = int(positions[:, 0].max())
    max_y = int(positions[:, 1].max())
    min_x = int(positions[:, 0].min())
    min_y = int(positions[:, 1].min())

    valid_starts = []
    for x in range(min_x, max_x - region_w + 2):
        for y in range(min_y, max_y - region_h + 2):
            # Check if all patches in this region exist
            all_present = True
            for dx in range(region_w):
                for dy in range(region_h):
                    if (x + dx, y + dy) not in pos_set:
                        all_present = False
                        break
                if not all_present:
                    break
            if all_present:
                valid_starts.append((x, y))

    if len(valid_starts) == 0:
        return []

    # Random sample
    n = min(n_queries, len(valid_starts))
    indices = np.random.choice(len(valid_starts), n, replace=False)
    return [valid_starts[i] for i in indices]


def extract_region_features(features, positions, start_x, start_y, region_w, region_h):
    """Extract features and positions for a specific region.

    Args:
        features: [N, D] all features
        positions: [N, 2] all positions
        start_x, start_y: top-left corner
        region_w, region_h: region dimensions
    Returns:
        region_features: [M, D]
        region_positions: [M, 2]
    """
    mask = (
        (positions[:, 0] >= start_x) &
        (positions[:, 0] < start_x + region_w) &
        (positions[:, 1] >= start_y) &
        (positions[:, 1] < start_y + region_h)
    )
    return features[mask], positions[mask]


def evaluate_region_retrieval(index, level, region_sizes=None,
                              n_queries=10, n_inits=10,
                              k_values=[1, 5, 10], top_m=200):
    """Run the complete region-level evaluation protocol.

    Args:
        index: WSIVectorIndex with built index
        level: magnification level to evaluate
        region_sizes: list of (w, h) tuples, default [(10,10), (20,10), (20,20)]
        n_queries: queries per region size (default 10)
        n_inits: random initializations per query (default 10)
        k_values: k values for Sim@k
        top_m: neighbors per patch (default 200)
    Returns:
        dict of results
    """
    if region_sizes is None:
        region_sizes = [(10, 10), (20, 10), (20, 20)]

    if level not in index.level_graphs:
        raise ValueError(f"Level {level} not in index")

    features = index.level_graphs[level]['features']
    positions = index.level_graphs[level]['positions']

    all_results = {}

    for region_w, region_h in region_sizes:
        size_key = f"{region_w}x{region_h}"
        print(f"\n--- Evaluating region size: {size_key} ---")

        # Find valid query regions
        query_starts = find_valid_query_regions(
            positions, region_w, region_h, n_queries=n_queries
        )

        if not query_starts:
            print(f"  No valid {size_key} regions found")
            all_results[size_key] = None
            continue

        print(f"  Found {len(query_starts)} valid query regions")

        # Collect Sim@k across queries and initializations
        sim_results = {f'Sim@{k}': [] for k in k_values}
        query_times = []

        for qi, (start_x, start_y) in enumerate(query_starts):
            # Extract query region
            q_features, q_positions = extract_region_features(
                features, positions, start_x, start_y, region_w, region_h
            )
            query_wh = (region_w, region_h)

            for init_i in range(n_inits):
                np.random.seed(qi * 1000 + init_i)

                start_time = time.time()

                # Search for all patches in query region
                region_results = index.search_region(
                    q_features, q_positions, top_k=top_m,
                    exploration_factor=2, use_gpu=True
                )

                query_time = time.time() - start_time
                query_times.append(query_time)

                # Aggregate into bounding boxes
                bboxes = index.aggregate_region_results(region_results)

                # Build retrieved region features for Sim@k evaluation
                retrieved_regions = []
                for bbox_info in bboxes[:max(k_values)]:
                    bbox_level = bbox_info['level']
                    bx0, by0, bx1, by1 = bbox_info['bbox']
                    r_features, r_positions = extract_region_features(
                        index.level_graphs[bbox_level]['features'],
                        index.level_graphs[bbox_level]['positions'],
                        bx0, by0, bx1 - bx0 + 1, by1 - by0 + 1
                    )
                    if len(r_features) > 0:
                        retrieved_regions.append({
                            'features': r_features,
                            'positions': r_positions,
                            'wh': (bx1 - bx0 + 1, by1 - by0 + 1)
                        })

                # Compute Sim@k
                sim_k = calculate_sim_at_k(
                    q_features, q_positions, query_wh,
                    retrieved_regions, k_values=k_values
                )
                for k in k_values:
                    sim_results[f'Sim@{k}'].append(sim_k[f'Sim@{k}'])

        # Average across queries and initializations
        size_result = {}
        for key, values in sim_results.items():
            size_result[key] = float(np.mean(values)) if values else 0.0
            size_result[f'{key}_std'] = float(np.std(values)) if values else 0.0

        size_result['avg_query_time_ms'] = float(np.mean(query_times) * 1000)
        size_result['n_queries'] = len(query_starts)
        size_result['n_inits'] = n_inits

        all_results[size_key] = size_result

        # Print
        for k in k_values:
            key = f'Sim@{k}'
            print(f"  {key}: {size_result[key]:.4f} +/- {size_result[f'{key}_std']:.4f}")
        print(f"  Avg query time: {size_result['avg_query_time_ms']:.2f} ms")

    return all_results


def evaluate_brute_force_comparison(index, level, n_queries=100, top_k=5):
    """Compare hierarchical search with brute-force baseline.

    Reports Acc@k and timing for both methods.
    """
    if level not in index.level_graphs:
        raise ValueError(f"Level {level} not in index")

    features = index.level_graphs[level]['features']
    n_total = features.shape[0]

    # Random query selection
    query_indices = np.random.choice(n_total, min(n_queries, n_total), replace=False)
    query_features = features[query_indices]

    # ----- Brute-force -----
    print("\n--- Brute-Force Search ---")
    bf_start = time.time()
    bf_indices, bf_sims = brute_force_batch_search(query_features, features, top_k=top_k)
    bf_time = (time.time() - bf_start) / len(query_indices) * 1000  # ms per query

    # ----- HIRES Search -----
    print("--- HIRES Hierarchical Search ---")
    hires_indices = np.zeros((len(query_indices), top_k), dtype=np.int64)
    hires_sims = np.zeros((len(query_indices), top_k))

    hires_start = time.time()
    for i, query in enumerate(query_features):
        results = index.search(query, top_k=top_k, use_gpu=True)
        for j, (lvl, idx, sim) in enumerate(results[:top_k]):
            global_idx = index.global_indices[lvl][0] + idx
            hires_indices[i, j] = global_idx
            hires_sims[i, j] = sim
    hires_time = (time.time() - hires_start) / len(query_indices) * 1000

    # ----- Compare -----
    # Recall: how many of brute-force top-k are in HIRES top-k
    recall_values = []
    for i in range(len(query_indices)):
        bf_set = set(bf_indices[i])
        hires_set = set(hires_indices[i])
        recall = len(bf_set & hires_set) / len(bf_set) if bf_set else 0
        recall_values.append(recall)

    results = {
        'brute_force_time_ms': bf_time,
        'hires_time_ms': hires_time,
        'speedup': bf_time / hires_time if hires_time > 0 else float('inf'),
        f'recall@{top_k}': float(np.mean(recall_values)),
        'n_queries': len(query_indices)
    }

    print(f"\nBrute-Force: {bf_time:.4f} ms/query")
    print(f"HIRES:       {hires_time:.4f} ms/query")
    print(f"Speedup:     {results['speedup']:.1f}x")
    print(f"Recall@{top_k}:   {results[f'recall@{top_k}']:.4f}")

    return results


# ================================================================
# Main
# ================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='HIRES Region-Level Evaluation')
    parser.add_argument('--index_path', type=str, required=True,
                        help='Path to saved WSIVectorIndex pickle')
    parser.add_argument('--level', type=int, default=1,
                        help='Magnification level to evaluate')
    parser.add_argument('--n_queries', type=int, default=10,
                        help='Number of query regions per size')
    parser.add_argument('--n_inits', type=int, default=10,
                        help='Number of random initializations per query')
    parser.add_argument('--output', type=str, default='region_eval_results.json',
                        help='Output file for results')
    args = parser.parse_args()

    # Load index
    print(f"Loading index from {args.index_path}...")
    index = WSIVectorIndex.load_index(args.index_path)
    print(f"  Levels: {list(index.level_graphs.keys())}")
    for level, data in index.level_graphs.items():
        print(f"  Level {level}: {len(data['features'])} patches")

    # Region-level evaluation
    print("\n" + "=" * 60)
    print("REGION-LEVEL EVALUATION")
    print("=" * 60)
    region_results = evaluate_region_retrieval(
        index, args.level,
        n_queries=args.n_queries,
        n_inits=args.n_inits
    )

    # Brute-force comparison
    print("\n" + "=" * 60)
    print("BRUTE-FORCE COMPARISON")
    print("=" * 60)
    bf_results = evaluate_brute_force_comparison(
        index, args.level, n_queries=100
    )

    # Save results
    output = {
        'region_evaluation': region_results,
        'brute_force_comparison': bf_results
    }
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")
