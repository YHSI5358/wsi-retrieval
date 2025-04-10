
import os
import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("...")
import json
import uuid
import cupy as cp
from cuvs.neighbors import cagra
from cuvs.neighbors import brute_force
from tqdm import tqdm
from datetime import datetime
from PIL import Image
from io import BytesIO
import requests

sys.path.append('/hpc2hdd/home/ysi538/retrieval')
sys.path.append(os.path.join(os.path.dirname(__file__), '../../utils'))
from encoder import WSI_Image_UNI_Encoder

import time

import numpy as np
from cuvs.neighbors import hnsw
import concurrent.futures
import multiprocessing
import cProfile
import pstats
import logging
from typing import List, Tuple, Dict, Any, Optional

UUID_NAMESPACE = uuid.NAMESPACE_URL



method_type = "cupy"

cuda_cache_dir_name = method_type + "_embeddings_level1+"
cuda_infos_dir_name = method_type + "_infos_level1+"
cuda_index_dir_name = method_type + "_index_level1+"


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Image2Image_Retriever_Rapid():
    def __init__(self, root_path: Optional[str] = None):
        self.image_client_name = "WSI_Region_Retrieval"
        self.root_path = root_path or os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
        self.embed_cache_path = os.path.join(self.root_path, "MDI_RAG_Image2Image_Research/data/embedding_cache")
        self.cupy_cache_dir = os.path.join(self.root_path, f"MDI_RAG_Image2Image_Research/data/{method_type}_embeddings_level1+")
        self.cupy_infos_dir = os.path.join(self.root_path, f"MDI_RAG_Image2Image_Research/data/{method_type}_infos_level1+")
        self.cupy_index_dir = os.path.join(self.root_path, f"MDI_RAG_Image2Image_Research/data/{method_type}_index_level1+")
        self._check_directories()
        self.image_encoder = WSI_Image_UNI_Encoder()
        self.image_names = os.listdir(self.embed_cache_path) if os.path.exists(self.embed_cache_path) else []
        self.index_list, self.infos_list = self.build_index_and_infos_list()

    def _check_directories(self):
        directories = [
            self.cupy_cache_dir,
            self.cupy_infos_dir,
            self.cupy_index_dir
        ]
        
        for directory in directories:
            if not os.path.exists(directory):
                logger.warning(f": {directory}")
                try:
                    os.makedirs(directory, exist_ok=True)
                    logger.info(f": {directory}")
                except Exception as e:
                    logger.error(f" {directory}: {str(e)}")
    
    def load_infos_and_cupy_embeddings(self, cupy_batch_files, infos_batch_files, begin=0, end=None):
        cupy_embeddings = []
        all_infos = []
        zip_list = list(zip(cupy_batch_files, infos_batch_files))

        for cupy_batch_file, infos_batch_file in zip_list[begin:end]:
            try:
                with open(os.path.join(self.cupy_cache_dir, cupy_batch_file), 'rb') as f:
                    batch_embeddings = cp.load(f) if CUDA_AVAILABLE else np.load(f)
                    
                    batch_embeddings = cp.asnumpy(batch_embeddings) if CUDA_AVAILABLE else batch_embeddings
                    cupy_embeddings.append(batch_embeddings)
                    
                    if CUDA_AVAILABLE:
                        del batch_embeddings
                        cp.get_default_memory_pool().free_all_blocks()
                with open(os.path.join(self.cupy_infos_dir, infos_batch_file), 'r') as f:
                    batch_infos = json.load(f)
                    all_infos += batch_infos
            except Exception as e:
                logger.error(f" {cupy_batch_file} / {infos_batch_file}: {str(e)}")

        
        cupy_embeddings = np.concatenate(cupy_embeddings, axis=0)
        return all_infos,cupy_embeddings

    def load_index_file(self, index_file):
        index = hnsw.load(os.path.join(self.cupy_index_dir, index_file), 1024, np.float32, "sqeuclidean")
        return index

    def load_info_file(self, info_file):
        with open(os.path.join(self.cupy_infos_dir, info_file), 'r') as f:
            infos = json.load(f)
        return infos

    
    def build_index_and_infos_list(self):
        cupy_batch_files = sorted([f for f in os.listdir(self.cupy_cache_dir) if f.startswith("cupy_embeddings_batch_")])
        infos_batch_files = sorted([f for f in os.listdir(self.cupy_infos_dir) if f.startswith("cupy_infos_batch_")])
        index_batch_files = sorted([f for f in os.listdir(self.cupy_index_dir) if f.startswith("cupy_index_batch_")])
        file_count = len(cupy_batch_files)
        print(f"File count: {file_count}")
        
        index_list = [None] * len(index_batch_files)
        infos_list = [None] * len(infos_batch_files)

        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            index_futures = {executor.submit(self.load_index_file, index_file): i for i, index_file in enumerate(index_batch_files)}
            info_futures = {executor.submit(self.load_info_file, info_file): i for i, info_file in enumerate(infos_batch_files)}

            for future in concurrent.futures.as_completed(index_futures):
                index = index_futures[future]
                index_list[index] = future.result()

            for future in concurrent.futures.as_completed(info_futures):
                index = info_futures[future]
                infos_list[index] = future.result()

        print("len(index_list):", len(index_list))
        print("len(infos_list):", len(infos_list))

        return index_list, infos_list
        
    
    def search(self, query_path, top_k=20):
        total_neighbors = []
        total_distances = []
        query_image = self.request_image(query_path)
        query_embedding = self.image_encoder.encode_image(query_image)
        
        query_embedding = np.array(query_embedding).astype('float32')

        
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        result_infos = []

        if method_type == "cupy":
            search_params = hnsw.SearchParams()

            time_cost = 0
            for index,hnsw_index in tqdm(enumerate(self.index_list), desc="Searching",ncols=100):
                begin_time = time.time()
                distances, neighbors = hnsw.search(search_params, hnsw_index, query_embedding, top_k)
                end_time = time.time()
                time_cost += end_time - begin_time
                
                neighbors = cp.asarray(neighbors).flatten().tolist()
                distances = cp.asarray(distances).flatten().tolist()
                for neighbor in neighbors:
                    result_infos.append(self.infos_list[index][neighbor])
                total_neighbors.extend(neighbors)
                total_distances.extend(distances)



        print("len(total_neighbors):", len(total_neighbors))
        print("len(total_distances):", len(total_distances))
        print("Time cost:", time_cost)


    
        total_distances, total_neighbors, result_infos = zip(*sorted(zip(total_distances, total_neighbors, result_infos)))
        return total_distances[:top_k], total_neighbors[:top_k], result_infos[:top_k]
    

    def search_multi_imgs(self, query_pathes, m,n, top_k=20):
        all_distances = []
        all_neighbors = []


        for query_path in query_pathes:

            distances, neighbors,result_infos = self.search(query_path, top_k)
            neighbors = cp.asarray(neighbors).flatten().tolist()

            all_neighbors.extend(neighbors)
            all_distances.extend(distances)

        connected_regions = self.get_combined_regions(result_infos)
        return all_distances, all_neighbors, connected_regions, result_infos

    def get_combined_regions(self, result_infos):

        search_info_list = []
        
        for info in result_infos:
            search_info = {}
            level = info.split("_")[-1].split(".")[0]
            w = info.split("_")[-3]
            h = info.split("_")[-2]
            x = info.split("_")[-5]
            y = info.split("_")[-4]
            id = info.split("_")[0]
            name = "_".join(info.split("_")[1:-5])
            search_info = {"id": id, "name": name, "x": x, "y": y, "w": w, "h": h, "level": level}
            search_info_list.append(search_info)

        def dfs(node, component, visited):
            
            visited[node] = True
            
            component.append(search_info_list[node])
            
            
            for neighbor in range(len(search_info_list)):
                if not visited[neighbor] and self.judge_if_connected(search_info_list[node], search_info_list[neighbor]):
                    
                    dfs(neighbor, component, visited)

        
        visited = [False] * len(search_info_list)
        components = []

        
        for i in range(len(search_info_list)):
            if not visited[i]:
                
                current_component = []
                
                dfs(i, current_component, visited)
                
                components.append(current_component)

        
        components = [component for component in components if len(component) > 1]

        return components

    def judge_if_connected(self, info1, info2):
        
        if info1["name"] != info2["name"]:
            return False
        if info1["level"] != info2["level"]:
            return False
        if int(info1["x"]) + int(info1["w"]) == int(info2["x"]) and int(info1["y"]) == int(info2["y"]):
            return True
        if int(info1["x"]) - int(info1["w"]) == int(info2["x"]) and int(info1["y"]) == int(info2["y"]):
            return True
        if int(info1["y"]) + int(info1["h"]) == int(info2["y"]) and int(info1["x"]) == int(info2["x"]):
            return True
        if int(info1["y"]) - int(info1["h"]) == int(info2["y"]) and int(info1["x"]) == int(info2["x"]):
            return True
        if int(info1["x"]) + int(info1["w"]) == int(info2["x"]) and int(info1["y"]) + int(info1["h"]) == int(info2["y"]):
            return True
        if int(info1["x"]) - int(info1["w"]) == int(info2["x"]) and int(info1["y"]) - int(info1["h"]) == int(info2["y"]):
            return True
        if int(info1["x"]) + int(info1["w"]) == int(info2["x"]) and int(info1["y"]) - int(info1["h"]) == int(info2["y"]):
            return True
        if int(info1["x"]) - int(info1["w"]) == int(info2["x"]) and int(info1["y"]) + int(info1["h"]) == int(info2["y"]):
            return True
        return False


    def request_image(self, query_img_path):
        if "http" in query_img_path:
            response = requests.get(query_img_path, verify=False)
            query_image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            query_image = Image.open(query_img_path).convert("RGB")
        return query_image
    
    def __del__(self):
        
        if hasattr(self, 'index_list'):
            for index in self.index_list:
                del index
        if cp is not None:
            cp.get_default_memory_pool().free_all_blocks()































if __name__ == "__main__":
    
    begin = time.time()
    builder = Image2Image_Retriever_Rapid()
    end = time.time()
    print(f"Time cost: {end-begin}")
    print("Finish building the image vector database.")

    
    query_img_path =  " metaservice/api/region/openslide/241183-21.tiff/6400/25344/256/256/1"
    
    
    m=1
    n=1
    begin = time.time()
    distances, neighbors, connected_regions,result_infos = builder.search_multi_imgs([query_img_path],m,n ,top_k=20)
    end = time.time()
    print(f"Time cost: {end-begin}")
    for info in result_infos:
        print(info)
        print("\n")
    


