import os
import json
import pickle
import numpy as np
import numba as nb
from tqdm import tqdm
import cupy as cp

def calculate_map5(neighbors, sub_ids, codes, ground_truth_codes, unique_subtypes):
    """ MAP@5 """
    ap_scores = []
    n_queries = neighbors.shape[0]
    
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
        
        
        ap = 0.0
        correct_at_k = 0
        
        for k in range(len(valid_codes)):
            if valid_codes[k] == true_code:
                correct_at_k += 1
                precision_at_k = correct_at_k / (k + 1)
                ap += precision_at_k
        
        
        if correct_at_k > 0:
            ap /= min(5, correct_at_k)
        else:
            ap = 0.0
        
        ap_scores.append(ap)
    
    return np.mean(ap_scores) if ap_scores else 0.0


root_path = '/hpc2hdd/home/ysi538/my_cuda_code/TCGA_slide_retrieval/' 

total_site = os.listdir(os.path.join(root_path, "total_for_site_foreground"))
total_site = [os.path.join(root_path, "total_for_site_foreground", site) for site in total_site]

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
    
    
    @nb.njit(parallel=True)
    def process_query(neighbors, sub_ids, codes, num_types):
        n = neighbors.shape[0]
        results = np.empty(n, dtype=np.int32)
        is_dismiss = np.zeros(n, dtype=np.bool_)
        
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
    
    
    map5 = calculate_map5(neighbors_np, sub_ids, codes, ground_truth_codes, unique_subtypes)
    
    
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
    
    
    with open(os.path.join(site, 'total_query_results.json'), 'w') as f:
        json.dump({
            'correct_count': correct_count,
            'incorrect_count': incorrect_count,
            'total_count': total_count,
            'valid_count': valid_count,  
            'dismiss_count': int(dismiss_count),
            'map5': float(map5)
        }, f)
    
    
    site_name = os.path.basename(site)
    print(f"\nSite: {site_name}")
    print(f"MAP@5: {map5:.4f}")
    print(f"Dismissed cases: {dismiss_count}")
    
    for st in unique_subtypes:
        total = total_count[st]
        valid = valid_count[st]
        
        if valid == 0:
            accuracy = 0.0
        else:
            accuracy = correct_count[st] / valid
        
        print(
            f"Subtype: {st}, "
            f"Correct: {correct_count[st]}, Incorrect: {incorrect_count[st]}, "
            f"Valid: {valid}, Accuracy: {accuracy:.4f}"
        )