"""
HIRES Evaluation Pipeline

Implements all evaluation metrics described in the paper:
- Slide-level: mMV@5, mAP@5 (Section 6.1)
- Region-level: Sim@k (Section 6.2)
- ROI Benchmark: Acc@k, MVAcc@5 (Section 6.2)
"""

import os
import json
import pickle
import numpy as np
import numba as nb
from tqdm import tqdm
import cupy as cp


# ================================================================
# Slide-Level Metrics
# ================================================================

def calculate_map5(neighbors, sub_ids, codes, ground_truth_codes, unique_subtypes):
    """Mean Average Precision at 5 (mAP@5).

    FIX: AP normalization changed from `min(5, correct_at_k)` to `5` (= k).
    Paper formula: AP@k(q) = (1/k) * sum_{j=1}^{k} Precision@j * I(y_hat_{q,j} in R_q)
    """
    ap_scores = []
    n_queries = neighbors.shape[0]

    for i in range(n_queries):
        true_code = ground_truth_codes[i]
        current_id = sub_ids[i]

        # Get top-5 valid neighbors (excluding same patient)
        valid_codes = []
        seen_ids = {current_id}

        for j in range(neighbors.shape[1]):
            neighbor_idx = neighbors[i, j]
            if sub_ids[neighbor_idx] not in seen_ids:
                seen_ids.add(sub_ids[neighbor_idx])
                valid_codes.append(codes[neighbor_idx])
                if len(valid_codes) == 5:
                    break

        # Compute AP@5: (1/k) * sum Precision@j * rel(j)
        ap = 0.0
        correct_at_k = 0

        for k in range(len(valid_codes)):
            if valid_codes[k] == true_code:
                correct_at_k += 1
                precision_at_k = correct_at_k / (k + 1)
                ap += precision_at_k

        # FIX: normalize by k=5, not by min(5, correct_at_k)
        ap /= 5

        ap_scores.append(ap)

    return np.mean(ap_scores) if ap_scores else 0.0


def calculate_mmv5(neighbors, sub_ids, codes, ground_truth_codes, unique_subtypes):
    """Mean Majority Vote at 5 (mMV@5).

    mMV@k = (1/N) * sum I[MV({y_hat_{q,j}}_{j=1}^k) == y_q]
    """
    n_queries = neighbors.shape[0]
    n_types = len(unique_subtypes)
    correct = 0
    valid_count = 0

    for i in range(n_queries):
        true_code = ground_truth_codes[i]
        current_id = sub_ids[i]

        valid_codes = []
        seen_ids = {current_id}

        for j in range(neighbors.shape[1]):
            neighbor_idx = neighbors[i, j]
            if sub_ids[neighbor_idx] not in seen_ids:
                seen_ids.add(sub_ids[neighbor_idx])
                valid_codes.append(codes[neighbor_idx])
                if len(valid_codes) == 5:
                    break

        if len(valid_codes) < 5:
            continue

        valid_count += 1
        # Majority vote
        counts = np.zeros(n_types, dtype=np.int32)
        for code in valid_codes:
            counts[code] += 1
        predicted = np.argmax(counts)
        if predicted == true_code:
            correct += 1

    return correct / valid_count if valid_count > 0 else 0.0


# ================================================================
# ROI Benchmark Metrics (TCGA Pan-Cancer)
# ================================================================

def calculate_acc_at_k(neighbors, sub_ids, codes, ground_truth_codes, k_values=[1, 3, 5]):
    """Accuracy at K: success rate if at least one top-k ROI shares same class label.

    Acc@k = (1/N) * sum I[exists j in top-k: y_hat_j == y_q]
    """
    results = {}
    n_queries = neighbors.shape[0]

    for k in k_values:
        correct = 0
        valid = 0

        for i in range(n_queries):
            true_code = ground_truth_codes[i]
            current_id = sub_ids[i]

            valid_codes = []
            seen_ids = {current_id}

            for j in range(neighbors.shape[1]):
                neighbor_idx = neighbors[i, j]
                if sub_ids[neighbor_idx] not in seen_ids:
                    seen_ids.add(sub_ids[neighbor_idx])
                    valid_codes.append(codes[neighbor_idx])
                    if len(valid_codes) == k:
                        break

            if len(valid_codes) < k:
                continue

            valid += 1
            if true_code in valid_codes:
                correct += 1

        results[f'Acc@{k}'] = correct / valid if valid > 0 else 0.0

    return results


def calculate_mvacc5(neighbors, sub_ids, codes, ground_truth_codes, n_types):
    """Majority Vote Accuracy at 5 (MVAcc@5).

    Success if majority class among top-5 matches query label.
    """
    n_queries = neighbors.shape[0]
    correct = 0
    valid = 0

    for i in range(n_queries):
        true_code = ground_truth_codes[i]
        current_id = sub_ids[i]

        valid_codes = []
        seen_ids = {current_id}

        for j in range(neighbors.shape[1]):
            neighbor_idx = neighbors[i, j]
            if sub_ids[neighbor_idx] not in seen_ids:
                seen_ids.add(sub_ids[neighbor_idx])
                valid_codes.append(codes[neighbor_idx])
                if len(valid_codes) == 5:
                    break

        if len(valid_codes) < 5:
            continue

        valid += 1
        counts = np.zeros(n_types, dtype=np.int32)
        for code in valid_codes:
            counts[code] += 1
        if np.argmax(counts) == true_code:
            correct += 1

    return correct / valid if valid > 0 else 0.0


# ================================================================
# Region-Level Metrics
# ================================================================

def calculate_sim_at_k(query_features, query_positions, query_wh,
                       retrieved_regions, k_values=[1, 5, 10]):
    """Similarity at top-k (Sim@k) as defined in the paper.

    Sim@k = (1/N) * sum_{i=1}^{N} (1/k) * sum_{j=1}^{k} Sim(r_phi_q^i, r_phi_j^i)

    Uses the region similarity with aspect ratio penalty alpha.

    Args:
        query_features: [N_patches, D] features of query region
        query_positions: [N_patches, 2] positions
        query_wh: (w, h) query region size in patches
        retrieved_regions: list of dicts, each with:
            'features': [M, D], 'positions': [M, 2], 'wh': (w, h)
        k_values: list of k values to evaluate
    Returns:
        dict of {k: sim_at_k}
    """
    from temp_index import region_similarity

    results = {}
    n_retrieved = len(retrieved_regions)

    for k in k_values:
        k_actual = min(k, n_retrieved)
        if k_actual == 0:
            results[f'Sim@{k}'] = 0.0
            continue

        sim_sum = 0.0
        for j in range(k_actual):
            region = retrieved_regions[j]
            sim = region_similarity(
                query_features, query_positions, query_wh,
                region['features'], region['positions'], region['wh']
            )
            sim_sum += sim

        results[f'Sim@{k}'] = sim_sum / k_actual

    return results


# ================================================================
# Brute-Force Baseline for Comparison
# ================================================================

def brute_force_search(query_features, all_features, top_k=5):
    """Brute-force cosine similarity search for baseline comparison.

    Args:
        query_features: [D] single query vector
        all_features: [N, D] all database features
        top_k: number of results
    Returns:
        indices: [top_k] indices of nearest neighbors
        similarities: [top_k] cosine similarities
    """
    # Cosine similarity
    query_norm = np.linalg.norm(query_features)
    if query_norm < 1e-8:
        return np.zeros(top_k, dtype=np.int64), np.zeros(top_k)

    all_norms = np.linalg.norm(all_features, axis=1)
    all_norms = np.maximum(all_norms, 1e-8)

    similarities = all_features @ query_features / (all_norms * query_norm)

    top_indices = np.argpartition(-similarities, top_k)[:top_k]
    top_indices = top_indices[np.argsort(-similarities[top_indices])]
    top_sims = similarities[top_indices]

    return top_indices, top_sims


def brute_force_batch_search(query_batch, all_features, top_k=5):
    """Batch brute-force search.

    Args:
        query_batch: [Q, D] query vectors
        all_features: [N, D] database features
        top_k: results per query
    Returns:
        all_indices: [Q, top_k]
        all_sims: [Q, top_k]
    """
    Q = query_batch.shape[0]
    all_indices = np.zeros((Q, top_k), dtype=np.int64)
    all_sims = np.zeros((Q, top_k))

    # Normalize all features once
    all_norms = np.linalg.norm(all_features, axis=1)
    all_norms = np.maximum(all_norms, 1e-8)
    all_features_normed = all_features / all_norms[:, None]

    for i in range(Q):
        q = query_batch[i]
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-8:
            continue
        q_normed = q / q_norm
        sims = all_features_normed @ q_normed
        top_idx = np.argpartition(-sims, top_k)[:top_k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        all_indices[i] = top_idx
        all_sims[i] = sims[top_idx]

    return all_indices, all_sims


# ================================================================
# Main Evaluation Script (Slide-Level)
# ================================================================

if __name__ == '__main__':
    root_path = '/hpc2hdd/home/ysi538/my_cuda_code/TCGA_slide_retrieval/'

    total_site = os.listdir(os.path.join(root_path, "total_for_site_foreground"))
    total_site = [os.path.join(root_path, "total_for_site_foreground", site) for site in total_site]

    all_results = {}

    for site in tqdm(total_site[:]):

        with open(os.path.join(site, 'total_query_results.pkl'), 'rb') as f:
            distances, neighbors = pickle.load(f)
        with open(os.path.join(site, 'total_patch_info.json'), 'r') as f:
            total_patch_info = json.load(f)

        sub_ids = np.array([p['sub_id'] for p in total_patch_info])
        subtypes = [p['subtype'] for p in total_patch_info]
        ground_truth = subtypes.copy()

        unique_subtypes = list(set(subtypes))
        subtype_to_code = {st: i for i, st in enumerate(unique_subtypes)}
        code_to_subtype = {i: st for i, st in enumerate(unique_subtypes)}
        codes = np.array([subtype_to_code[st] for st in subtypes], dtype=np.int32)
        ground_truth_codes = codes.copy()

        neighbors_np = cp.asnumpy(neighbors)

        # Compute all metrics
        map5 = calculate_map5(neighbors_np, sub_ids, codes, ground_truth_codes, unique_subtypes)
        mmv5 = calculate_mmv5(neighbors_np, sub_ids, codes, ground_truth_codes, unique_subtypes)
        acc_results = calculate_acc_at_k(neighbors_np, sub_ids, codes, ground_truth_codes,
                                          k_values=[1, 3, 5])
        mvacc5 = calculate_mvacc5(neighbors_np, sub_ids, codes, ground_truth_codes,
                                   len(unique_subtypes))

        # Per-subtype accuracy (majority vote)
        @nb.njit(parallel=True)
        def process_query(neighbors, sub_ids, codes, num_types):
            n = neighbors.shape[0]
            results = np.empty(n, dtype=np.int32)
            is_dismiss = np.zeros(n, dtype=nb.boolean)

            for i in nb.prange(n):
                current_id = sub_ids[i]
                valid_codes = []

                for j in range(neighbors.shape[1]):
                    neighbor_idx = neighbors[i, j]
                    if sub_ids[neighbor_idx] != current_id:
                        valid_codes.append(codes[neighbor_idx])
                        if len(valid_codes) == 5:
                            break

                if len(valid_codes) < 5:
                    results[i] = -999
                    is_dismiss[i] = True
                else:
                    counts = np.zeros(num_types, dtype=np.int32)
                    for code in valid_codes:
                        counts[code] += 1
                    results[i] = np.argmax(counts)

            return results, is_dismiss

        query_results_codes, is_dismiss = process_query(
            neighbors_np, sub_ids, codes, len(unique_subtypes)
        )
        dismiss_count = np.sum(is_dismiss)

        query_results = []
        for code, dismissed in zip(query_results_codes, is_dismiss):
            if dismissed:
                query_results.append(None)
            else:
                query_results.append(code_to_subtype[code])

        correct_count = {st: 0 for st in unique_subtypes}
        incorrect_count = {st: 0 for st in unique_subtypes}
        valid_count = {st: 0 for st in unique_subtypes}

        for q, gt in zip(query_results, ground_truth):
            if q is None:
                correct_count[gt] += 1
                continue
            valid_count[gt] += 1
            if q == gt:
                correct_count[gt] += 1
            else:
                incorrect_count[gt] += 1

        total_count = {
            st: correct_count[st] + incorrect_count[st]
            for st in unique_subtypes
        }

        # Save all metrics
        site_results = {
            'correct_count': correct_count,
            'incorrect_count': incorrect_count,
            'total_count': total_count,
            'valid_count': valid_count,
            'dismiss_count': int(dismiss_count),
            'map5': float(map5),
            'mmv5': float(mmv5),
            'acc_at_k': acc_results,
            'mvacc5': float(mvacc5)
        }

        with open(os.path.join(site, 'total_query_results.json'), 'w') as f:
            json.dump(site_results, f, indent=2)

        all_results[os.path.basename(site)] = site_results

        # Print results
        site_name = os.path.basename(site)
        print(f"\nSite: {site_name}")
        print(f"  mAP@5:    {map5:.4f}")
        print(f"  mMV@5:    {mmv5:.4f}")
        for k, v in acc_results.items():
            print(f"  {k}:     {v:.4f}")
        print(f"  MVAcc@5:  {mvacc5:.4f}")
        print(f"  Dismissed: {dismiss_count}")

        for st in unique_subtypes:
            total = total_count[st]
            valid = valid_count[st]
            accuracy = correct_count[st] / valid if valid > 0 else 0.0
            print(f"  Subtype: {st}, Correct: {correct_count[st]}, "
                  f"Incorrect: {incorrect_count[st]}, Accuracy: {accuracy:.4f}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY ACROSS ALL SITES")
    print("=" * 60)
    avg_map5 = np.mean([r['map5'] for r in all_results.values()])
    avg_mmv5 = np.mean([r['mmv5'] for r in all_results.values()])
    print(f"Average mAP@5:   {avg_map5:.4f}")
    print(f"Average mMV@5:   {avg_mmv5:.4f}")
