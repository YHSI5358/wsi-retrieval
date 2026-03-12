#%%
import os
import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("...")
import json
import uuid
import cupy as cp
from tqdm import tqdm
from datetime import datetime
from PIL import Image
from io import BytesIO
import requests
sys.path.append('/hpc2hdd/home/ysi538/retrieval')
from MDI_RAG_Image2Image_Research.src.utils.encoder import WSI_Image_UNI_Encoder
import time

import numpy as np
from cuvs.neighbors import hnsw
import concurrent.futures
import multiprocessing
from flask import Flask, request, jsonify

# Import hierarchical index
sys.path.append(os.path.join(os.path.dirname(__file__), 'TCGA_slide_retrieval/code'))
from temp_index import WSIVectorIndex
from utils import split_img_fun, get_combined_regions_cpu, compute_region_score, cosine_similarity


app = Flask(__name__)

UUID_NAMESPACE = uuid.NAMESPACE_URL

#%%
# Search method: "hires" (paper's hierarchical index) or "hnsw" (cUVS fallback)
SEARCH_METHOD = os.environ.get("SEARCH_METHOD", "hires")

# Paths
HIRES_INDEX_PATH = os.environ.get(
    "HIRES_INDEX_PATH",
    "/hpc2ssd/JH_DATA/spooler/ysi538/hires_index.pkl"
)
HNSW_INDEX_PATH = os.environ.get(
    "HNSW_INDEX_PATH",
    "/hpc2ssd/JH_DATA/spooler/ysi538/cupy_index_batch_0.bin"
)
HNSW_INFOS_PATH = os.environ.get(
    "HNSW_INFOS_PATH",
    "/hpc2ssd/JH_DATA/spooler/ysi538/cupy_infos_batch_0.json"
)


class Image2Image_Retriever_HIRES():
    """Retriever using the paper's hierarchical index with CUDA kernels."""

    def __init__(self):
        self.image_client_name = "WSI_Region_Retrieval"
        self.image_encoder = WSI_Image_UNI_Encoder()

        print(f"Loading HIRES hierarchical index from {HIRES_INDEX_PATH}...")
        self.hires_index = WSIVectorIndex.load_index(HIRES_INDEX_PATH)
        print(f"  Levels: {list(self.hires_index.level_graphs.keys())}")
        for level, data in self.hires_index.level_graphs.items():
            print(f"  Level {level}: {len(data['features'])} patches")

        # Load info mapping (global_idx -> info string)
        with open(HNSW_INFOS_PATH, 'r') as f:
            self.infos = json.load(f)

    def search(self, query_path, top_k=20):
        query_image = self.request_image(query_path)
        query_embedding = self.image_encoder.encode_image(query_image)
        query_embedding = np.array(query_embedding).astype('float32')

        begin_time = time.time()
        # Use hierarchical index GPU search (CUDA kernels)
        results = self.hires_index.search_gpu(query_embedding, top_m=top_k)
        time_cost = time.time() - begin_time

        print(f"HIRES search: {len(results)} results, {time_cost:.4f}s")

        # Convert results to info strings
        result_infos = []
        similarities = []
        neighbor_indices = []

        for level, idx, sim in results[:top_k]:
            global_idx = self.hires_index.global_indices[level][0] + idx
            if global_idx < len(self.infos):
                result_infos.append(self.infos[global_idx])
                similarities.append(sim)
                neighbor_indices.append(global_idx)

        return similarities, neighbor_indices, result_infos

    def search_multi_imgs(self, query_pathes, m, n, top_k=20):
        """Search with multiple query images (region query).

        Uses batch GPU search via CUDA kernels for all patches simultaneously.
        """
        all_similarities = []
        all_neighbors = []
        all_result_infos = []

        # Collect all patch features
        all_patch_features = []
        all_patch_positions = []

        for query_path in query_pathes:
            query_image = self.request_image(query_path)
            patches = split_img_fun(query_image, None)
            for i, patch in enumerate(patches):
                embedding = self.image_encoder.encode_image(patch)
                all_patch_features.append(np.array(embedding).astype('float32'))
                all_patch_positions.append([i // n, i % n])

        if all_patch_features:
            features_batch = np.array(all_patch_features)
            positions_batch = np.array(all_patch_positions)

            # Single batch GPU search for all patches
            region_results = self.hires_index.search_region_gpu(
                features_batch, positions_batch, top_m=top_k
            )

            # Aggregate into connected regions with area-normalized scoring
            bboxes = self.hires_index.aggregate_region_results(region_results)

            # Collect all result infos for connected region detection
            for pos, results in region_results.items():
                for level, idx, sim in results[:top_k]:
                    global_idx = self.hires_index.global_indices[level][0] + idx
                    if global_idx < len(self.infos):
                        all_result_infos.append(self.infos[global_idx])
                        all_similarities.append(sim)
                        all_neighbors.append(global_idx)

        # Connected region detection (DFS)
        connected_regions = get_combined_regions_cpu(all_result_infos) if all_result_infos else []

        return all_similarities, all_neighbors, connected_regions, all_result_infos

    def request_image(self, query_img_path):
        if "http" in query_img_path:
            response = requests.get(query_img_path, verify=False)
            query_image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            query_image = Image.open(query_img_path).convert("RGB")
        return query_image


class Image2Image_Retriever_HNSW():
    """Fallback retriever using cUVS HNSW (original production system)."""

    def __init__(self):
        self.image_client_name = "WSI_Region_Retrieval"
        self.image_encoder = WSI_Image_UNI_Encoder()
        # FIX: use cosine metric instead of sqeuclidean
        self.index = hnsw.load(HNSW_INDEX_PATH, 1024, np.float32, "cosine")
        with open(HNSW_INFOS_PATH, 'r') as f:
            self.infos = json.load(f)

    def search(self, query_path, top_k=20):
        query_image = self.request_image(query_path)
        query_embedding = self.image_encoder.encode_image(query_image)
        query_embedding = np.array(query_embedding).astype('float32')
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        search_params = hnsw.SearchParams()
        begin_time = time.time()
        distances, neighbors = hnsw.search(search_params, self.index, query_embedding, top_k)
        time_cost = time.time() - begin_time

        neighbors = cp.asarray(neighbors).flatten().tolist()
        distances = cp.asarray(distances).flatten().tolist()
        result_infos = [self.infos[n] for n in neighbors]

        print(f"HNSW search: {len(neighbors)} results, {time_cost:.4f}s")

        distances, neighbors, result_infos = zip(*sorted(zip(distances, neighbors, result_infos)))
        return list(distances[:top_k]), list(neighbors[:top_k]), list(result_infos[:top_k])

    def request_image(self, query_img_path):
        if "http" in query_img_path:
            response = requests.get(query_img_path, verify=False)
            query_image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            query_image = Image.open(query_img_path).convert("RGB")
        return query_image

    def __del__(self):
        if cp is not None:
            cp.get_default_memory_pool().free_all_blocks()


image2image_retrieval = None

def get_image2image_retriever():
    global image2image_retrieval
    if image2image_retrieval is None:
        print(f"============== Initializing {SEARCH_METHOD} retriever ==============")
        begin_time = datetime.now()

        if SEARCH_METHOD == "hires" and os.path.exists(HIRES_INDEX_PATH):
            image2image_retrieval = Image2Image_Retriever_HIRES()
        else:
            if SEARCH_METHOD == "hires":
                print(f"WARNING: HIRES index not found at {HIRES_INDEX_PATH}, falling back to HNSW")
            image2image_retrieval = Image2Image_Retriever_HNSW()

        end_time = datetime.now()
        print(f"============== Initialized in {end_time - begin_time} ==============")
    return image2image_retrieval


@app.route('/test', methods=['POST'])
def process_image2image_retrieval():
    query_img_path = request.json.get('query_img_path')
    top_k = request.json.get('top_k', 20)

    image2image_retriever = get_image2image_retriever()
    distances, neighbors, results = image2image_retriever.search(query_img_path, top_k)

    retrieved_images_information = [{
        "wsi_level": node.split("_")[-1].split(".")[0],
        "metadata_position": (node.split("_")[-5], node.split("_")[-4]),
        "patch_size": (node.split("_")[-3], node.split("_")[-2]),
        "source_wsi_name": "_".join(node.split("_")[:-5]),
    } for node in results]

    return jsonify({
        'answer': "Image2Image Retrieval",
        'search_method': SEARCH_METHOD,
        'retrieved_images_information': retrieved_images_information
    })


if __name__ == "__main__":
    #%%
    get_image2image_retriever()
    app.run(debug=True, host='0.0.0.0', port="9876", use_reloader=False)
