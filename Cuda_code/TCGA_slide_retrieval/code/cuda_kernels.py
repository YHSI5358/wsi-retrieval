"""
HIRES GPU-Optimized Search Kernels

Implements the two core GPU optimizations described in the HIRES paper:
1. Warp-Centric Cosine Similarity Computation (Section 5.2)
   - Each 32-thread warp splits into 4 teams of 8 threads
   - Each team computes cosine similarity for one candidate
   - 1024-D features partitioned: 128 dims per thread
   - Intra-team reduction via __shfl_down_sync

2. Forgettable Hash Tables (Section 5.3)
   - 512-entry open-addressing hash table in shared memory (2KB)
   - Tracks visited nodes to prevent redundant computation
   - Reset every 4 search iterations to maintain <3% collision rate
"""

import cupy as cp
import numpy as np

# ============================================================
# CUDA Kernel Source Code
# ============================================================

HIRES_SEARCH_KERNEL = r"""
extern "C" {

// ============================================================
// Forgettable Hash Table: Open-Addressing in Shared Memory
// ============================================================

__device__ __forceinline__ unsigned int hash_func(int key, int table_size) {
    // Knuth multiplicative hash for better distribution
    return ((unsigned int)key * 2654435761u) % (unsigned int)table_size;
}

__device__ bool hash_insert(int* table, int table_size, int key) {
    unsigned int h = hash_func(key, table_size);
    for (int probe = 0; probe < 32; probe++) {
        unsigned int slot = (h + probe) % (unsigned int)table_size;
        int old = atomicCAS(&table[slot], -1, key);
        if (old == -1) return true;   // inserted (was new)
        if (old == key) return false;  // already existed
    }
    return true;  // max probes exceeded, treat as new (rare)
}

__device__ bool hash_lookup(int* table, int table_size, int key) {
    unsigned int h = hash_func(key, table_size);
    for (int probe = 0; probe < 32; probe++) {
        unsigned int slot = (h + probe) % (unsigned int)table_size;
        int val = table[slot];
        if (val == key) return true;   // found
        if (val == -1)  return false;  // empty slot -> not found
    }
    return false;  // max probes, not found
}

// ============================================================
// Main HIRES Search Kernel
//
// Grid:  Q blocks  (one per query vector)
// Block: 256 threads = 8 warps = 32 teams (4 teams/warp, 8 threads/team)
//
// Shared memory: hash_table[512] + counters[4]
// Per-query global memory: result buffers + staging buffers
// ============================================================

__global__ void hires_search_kernel(
    const float* __restrict__ features,       // [N, D]  all node features
    const int*   __restrict__ graph,          // [N, degree] adjacency (uniform degree)
    const float* __restrict__ queries,        // [Q, D]  query features
    float*       result_sims,                 // [Q, top_m] output similarities
    int*         result_idxs,                 // [Q, top_m] output node indices
    float*       staging_sims,                // [Q, max_staging] temp buffer
    int*         staging_idxs,                // [Q, max_staging] temp buffer
    const int N,
    const int D,
    const int degree,
    const int top_m,
    const int max_staging,
    const int max_iters,
    const int hash_table_size,
    const int hash_reset_interval,
    const int exploration_p,
    const unsigned long long seed
) {
    // ---- Thread / warp / team identification ----
    const int qid = blockIdx.x;
    const int tid = threadIdx.x;
    const int n_threads = blockDim.x;             // 256
    const int warp_id   = tid / 32;
    const int lane_id   = tid % 32;
    const int team_id   = lane_id / 8;            // 0-3 within warp
    const int team_lane = lane_id % 8;            // 0-7 within team
    const int n_warps   = n_threads / 32;         // 8
    const int n_teams   = n_warps * 4;            // 32
    const int global_team_id = warp_id * 4 + team_id;

    const int dims_per_thread = D / 8;            // 128
    const int dim_offset = team_lane * dims_per_thread;

    // ---- Shared memory layout ----
    extern __shared__ int smem[];
    int* hash_table       = smem;                            // [hash_table_size]
    int* shared_n_staging = smem + hash_table_size;          // [1]
    int* shared_converged = smem + hash_table_size + 1;      // [1]
    int* shared_n_results = smem + hash_table_size + 2;      // [1]

    // ---- Per-query global memory pointers ----
    float* my_result_sims  = result_sims  + qid * top_m;
    int*   my_result_idxs  = result_idxs  + qid * top_m;
    float* my_staging_sims = staging_sims + qid * max_staging;
    int*   my_staging_idxs = staging_idxs + qid * max_staging;

    // ---- Initialize shared memory ----
    for (int i = tid; i < hash_table_size; i += n_threads) {
        hash_table[i] = -1;
    }
    if (tid == 0) {
        shared_n_staging[0] = 0;
        shared_converged[0] = 0;
        shared_n_results[0] = 0;
    }
    // ---- Initialize results ----
    for (int i = tid; i < top_m; i += n_threads) {
        my_result_sims[i] = -2.0f;  // cosine sim range [-1, 1]
        my_result_idxs[i] = -1;
    }
    __syncthreads();

    // ---- Precompute query norm (partial per thread, reduce within team) ----
    const float* query = queries + qid * D;
    float qnorm_partial = 0.0f;
    for (int d = 0; d < dims_per_thread; d++) {
        float q = query[dim_offset + d];
        qnorm_partial += q * q;
    }
    for (int offset = 4; offset >= 1; offset /= 2) {
        qnorm_partial += __shfl_down_sync(0xFFFFFFFF, qnorm_partial, offset, 8);
    }
    float qnorm = sqrtf(__shfl_sync(0xFFFFFFFF, qnorm_partial, 0, 8));

    // ================================================================
    // PHASE 1: Random Initialization
    // Each team picks one random node, computes cosine similarity
    // ================================================================
    int init_count = exploration_p * degree;   // e.g. 2*64 = 128
    for (int i = global_team_id; i < init_count; i += n_teams) {
        // Generate pseudo-random node index
        int node = -1;
        int is_new = 0;
        if (team_lane == 0) {
            unsigned long long r = seed + (unsigned long long)qid * 100003ULL + (unsigned long long)i;
            r = r * 6364136223846793005ULL + 1442695040888963407ULL;
            node = (int)((r >> 16) % (unsigned long long)N);
            // Check and insert into hash table
            bool not_in_hash = !hash_lookup(hash_table, hash_table_size, node);
            if (not_in_hash) {
                hash_insert(hash_table, hash_table_size, node);
            }
            is_new = not_in_hash ? 1 : 0;
        }
        // Broadcast node and is_new to team
        node   = __shfl_sync(0xFFFFFFFF, node,   0, 8);
        is_new = __shfl_sync(0xFFFFFFFF, is_new, 0, 8);

        if (!is_new || node < 0 || node >= N) continue;

        // ---- Warp-centric cosine similarity ----
        const float* cand_feat = features + node * D;
        float dot = 0.0f, nc = 0.0f;
        for (int d = 0; d < dims_per_thread; d++) {
            float q = query[dim_offset + d];
            float c = cand_feat[dim_offset + d];
            dot += q * c;
            nc  += c * c;
        }
        // Intra-team reduction (width=8)
        for (int offset = 4; offset >= 1; offset /= 2) {
            dot += __shfl_down_sync(0xFFFFFFFF, dot, offset, 8);
            nc  += __shfl_down_sync(0xFFFFFFFF, nc,  offset, 8);
        }
        // Team leader writes to staging
        if (team_lane == 0) {
            float sim = dot / (qnorm * sqrtf(nc) + 1e-8f);
            int pos = atomicAdd(shared_n_staging, 1);
            if (pos < max_staging) {
                my_staging_sims[pos] = sim;
                my_staging_idxs[pos] = node;
            }
        }
    }
    __syncthreads();

    // ---- Thread 0: merge staging into results ----
    if (tid == 0) {
        int n_new = shared_n_staging[0];
        if (n_new > max_staging) n_new = max_staging;
        int n_res = 0;
        // Insert each staging entry into sorted results (descending sim)
        for (int s = 0; s < n_new; s++) {
            float new_sim = my_staging_sims[s];
            int   new_idx = my_staging_idxs[s];
            // Find insertion position
            int pos = n_res;
            while (pos > 0 && my_result_sims[pos - 1] < new_sim) {
                if (pos < top_m) {
                    my_result_sims[pos] = my_result_sims[pos - 1];
                    my_result_idxs[pos] = my_result_idxs[pos - 1];
                }
                pos--;
            }
            if (pos < top_m) {
                my_result_sims[pos] = new_sim;
                my_result_idxs[pos] = new_idx;
                if (n_res < top_m) n_res++;
            }
        }
        shared_n_results[0] = n_res;
        shared_n_staging[0] = 0;
    }
    __syncthreads();

    // ================================================================
    // PHASE 2: Iterative Graph Search
    // ================================================================
    for (int iter = 0; iter < max_iters; iter++) {

        // ---- Forgettable hash table: reset every hash_reset_interval iters ----
        if (iter > 0 && (iter % hash_reset_interval) == 0) {
            for (int i = tid; i < hash_table_size; i += n_threads) {
                hash_table[i] = -1;
            }
            __syncthreads();
            // Re-insert current result nodes
            if (tid == 0) {
                int n_res = shared_n_results[0];
                for (int i = 0; i < n_res; i++) {
                    if (my_result_idxs[i] >= 0) {
                        hash_insert(hash_table, hash_table_size, my_result_idxs[i]);
                    }
                }
            }
            __syncthreads();
        }

        // ---- Explore neighbors of top-p results ----
        int n_seeds = shared_n_results[0];
        if (n_seeds > exploration_p) n_seeds = exploration_p;
        int total_neighbors = n_seeds * degree;

        for (int ni = global_team_id; ni < total_neighbors; ni += n_teams) {
            int seed_i   = ni / degree;
            int neigh_j  = ni % degree;

            // Read seed node from results
            int seed_node = my_result_idxs[seed_i];
            if (seed_node < 0) continue;

            // Read neighbor from graph adjacency
            int neighbor = graph[seed_node * degree + neigh_j];

            // Broadcast neighbor to team
            int neighbor_bc = __shfl_sync(0xFFFFFFFF, neighbor, 0, 8);
            // Actually each thread can read neighbor independently since it's the same
            // But team_lane != 0 might not have the right seed_node. Let me fix:
            // All threads in the team should use the same neighbor.
            // Since ni is computed from global_team_id which is per-team, all threads in
            // the same team have the same ni, hence same seed_i and neigh_j.
            // So all threads can read neighbor independently.
            if (neighbor < 0 || neighbor >= N) continue;

            // Hash check (team leader only)
            int is_new = 0;
            if (team_lane == 0) {
                bool not_in_hash = !hash_lookup(hash_table, hash_table_size, neighbor);
                if (not_in_hash) {
                    hash_insert(hash_table, hash_table_size, neighbor);
                }
                is_new = not_in_hash ? 1 : 0;
            }
            is_new = __shfl_sync(0xFFFFFFFF, is_new, 0, 8);
            if (!is_new) continue;

            // ---- Warp-centric cosine similarity ----
            const float* cand_feat = features + neighbor * D;
            float dot = 0.0f, nc = 0.0f;
            for (int d = 0; d < dims_per_thread; d++) {
                float q = query[dim_offset + d];
                float c = cand_feat[dim_offset + d];
                dot += q * c;
                nc  += c * c;
            }
            for (int offset = 4; offset >= 1; offset /= 2) {
                dot += __shfl_down_sync(0xFFFFFFFF, dot, offset, 8);
                nc  += __shfl_down_sync(0xFFFFFFFF, nc,  offset, 8);
            }
            if (team_lane == 0) {
                float sim = dot / (qnorm * sqrtf(nc) + 1e-8f);
                int pos = atomicAdd(shared_n_staging, 1);
                if (pos < max_staging) {
                    my_staging_sims[pos] = sim;
                    my_staging_idxs[pos] = neighbor;
                }
            }
        }
        __syncthreads();

        // ---- Thread 0: merge staging into results ----
        if (tid == 0) {
            int n_new = shared_n_staging[0];
            if (n_new > max_staging) n_new = max_staging;
            int n_res = shared_n_results[0];
            bool any_updated = false;

            for (int s = 0; s < n_new; s++) {
                float new_sim = my_staging_sims[s];
                int   new_idx = my_staging_idxs[s];

                // Skip if worse than worst result and buffer is full
                if (n_res >= top_m && new_sim <= my_result_sims[n_res - 1]) continue;

                // Insertion sort (descending similarity)
                int pos = n_res - 1;
                if (n_res < top_m) pos = n_res;
                while (pos > 0 && my_result_sims[pos - 1] < new_sim) {
                    if (pos < top_m) {
                        my_result_sims[pos] = my_result_sims[pos - 1];
                        my_result_idxs[pos] = my_result_idxs[pos - 1];
                    }
                    pos--;
                }
                if (pos < top_m) {
                    my_result_sims[pos] = new_sim;
                    my_result_idxs[pos] = new_idx;
                    if (n_res < top_m) n_res++;
                    any_updated = true;
                }
            }
            shared_n_results[0] = n_res;
            shared_n_staging[0] = 0;
            shared_converged[0] = any_updated ? 0 : 1;
        }
        __syncthreads();

        if (shared_converged[0]) break;
    }

    // ---- Write final results to output ----
    for (int i = tid; i < top_m; i += n_threads) {
        // result_sims/result_idxs are already my_result_sims/my_result_idxs
        // (they point to the same memory), so no copy needed
    }
}

}  // extern "C"
"""


class HiresGPUSearcher:
    """
    Python wrapper for the HIRES GPU search kernel.

    Manages GPU memory allocation and kernel launch for
    warp-centric graph search with forgettable hash tables.
    """

    def __init__(self, features_np, adjacency_np,
                 degree=64, feature_dim=1024,
                 hash_table_size=512, hash_reset_interval=4,
                 max_iters=20, exploration_p=2, top_m=200):
        """
        Args:
            features_np: numpy array [N, D] of node features (float32)
            adjacency_np: numpy array [N, degree] of neighbor indices (int32)
            degree: uniform graph degree (default 64)
            feature_dim: feature dimensionality (default 1024)
            hash_table_size: forgettable hash table entries (default 512)
            hash_reset_interval: reset hash table every N iterations (default 4)
            max_iters: maximum search iterations (default 20)
            exploration_p: number of seed nodes to explore per iteration (default 2)
            top_m: number of results per query (default 200)
        """
        self.N = features_np.shape[0]
        self.D = feature_dim
        self.degree = degree
        self.top_m = top_m
        self.max_staging = exploration_p * degree * 2  # buffer margin
        self.max_iters = max_iters
        self.hash_table_size = hash_table_size
        self.hash_reset_interval = hash_reset_interval
        self.exploration_p = exploration_p

        # Transfer data to GPU
        self.features_gpu = cp.asarray(features_np, dtype=cp.float32)
        self.adjacency_gpu = cp.asarray(adjacency_np, dtype=cp.int32)

        # Compile kernel
        self.kernel = cp.RawKernel(HIRES_SEARCH_KERNEL, 'hires_search_kernel')

    def search(self, query_features, top_m=None):
        """
        Search for nearest neighbors of query features.

        Args:
            query_features: numpy array [Q, D] of query vectors (float32)
            top_m: override for number of results per query

        Returns:
            similarities: numpy array [Q, top_m] cosine similarities (descending)
            indices: numpy array [Q, top_m] node indices
        """
        if top_m is None:
            top_m = self.top_m

        Q = query_features.shape[0]
        queries_gpu = cp.asarray(query_features, dtype=cp.float32)

        # Allocate output buffers
        result_sims = cp.full((Q, top_m), -2.0, dtype=cp.float32)
        result_idxs = cp.full((Q, top_m), -1, dtype=cp.int32)
        staging_sims = cp.zeros((Q, self.max_staging), dtype=cp.float32)
        staging_idxs = cp.zeros((Q, self.max_staging), dtype=cp.int32)

        # Kernel launch configuration
        threads_per_block = 256  # 8 warps = 32 teams
        grid = (Q,)
        block = (threads_per_block,)
        # Shared memory: hash_table[512] + 3 counters
        shared_mem_bytes = (self.hash_table_size + 3) * 4

        seed = np.random.randint(0, 2**62, dtype=np.uint64)

        self.kernel(
            grid, block,
            (
                self.features_gpu,
                self.adjacency_gpu,
                queries_gpu,
                result_sims,
                result_idxs,
                staging_sims,
                staging_idxs,
                np.int32(self.N),
                np.int32(self.D),
                np.int32(self.degree),
                np.int32(top_m),
                np.int32(self.max_staging),
                np.int32(self.max_iters),
                np.int32(self.hash_table_size),
                np.int32(self.hash_reset_interval),
                np.int32(self.exploration_p),
                np.uint64(seed),
            ),
            shared_mem=shared_mem_bytes
        )

        cp.cuda.Stream.null.synchronize()

        return cp.asnumpy(result_sims), cp.asnumpy(result_idxs)

    def search_single(self, query_feature, top_m=None):
        """Search with a single query vector [D]."""
        query_batch = query_feature.reshape(1, -1)
        sims, idxs = self.search(query_batch, top_m=top_m)
        return sims[0], idxs[0]
