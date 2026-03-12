#%%
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
#  sys.path
sys.path.append('/hpc2hdd/home/ysi538/retrieval')
sys.path.append(os.path.join(os.path.dirname(__file__), '../../utils'))
from encoder import WSI_Image_UNI_Encoder
# from MDI_RAG_Image2Image_Research.src.utils.encoder import WSI_Image_UNI_Encoder
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from cuvs.neighbors import hnsw


UUID_NAMESPACE = uuid.NAMESPACE_URL

#%%
class Image2Image_Retriever_Rapid():
    def __init__(self):
        root_path = "/hpc2hdd/home/ysi538/retrieval/"
        self.embed_cache_path = root_path + "MDI_RAG_Image2Image_Research/data/embedding_cache"
        self.cupy_cache_dir = root_path + "MDI_RAG_Image2Image_Research/data/cupy_embeddings_level1+"
        self.cupy_infos_dir = root_path + "MDI_RAG_Image2Image_Research/data/cupy_infos_level1+"
        self.image_encoder = WSI_Image_UNI_Encoder()
        self.image_names = os.listdir(self.embed_cache_path)
        self.index_list, self.infos_list = self.build_index_and_infos_list()

    def load_infos_and_cupy_embeddings(self, cupy_batch_file, infos_batch_file):
        with open(os.path.join(self.cupy_cache_dir, cupy_batch_file), 'rb') as f:
            batch_embeddings = cp.load(f)
            batch_embeddings = cp.asnumpy(batch_embeddings)
        with open(os.path.join(self.cupy_infos_dir, infos_batch_file), 'r') as f:
            batch_infos = json.load(f)
        return batch_infos, batch_embeddings

    def build_index_and_infos_list(self):
        cupy_batch_files = sorted([f for f in os.listdir(self.cupy_cache_dir) if f.startswith("cupy_embeddings_batch_")])
        infos_batch_files = sorted([f for f in os.listdir(self.cupy_infos_dir) if f.startswith("cupy_infos_batch_")])
        file_count = len(cupy_batch_files)
        print(f"File count: {file_count}")

        index_list = []
        infos_list = []
        step = 2

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            for i in range(0, file_count, step):
                futures.append(executor.submit(self.load_and_build_index, cupy_batch_files[i:i+step], infos_batch_files[i:i+step]))

            for future in tqdm(as_completed(futures), total=len(futures), desc="Building index and infos list", ncols=100):
                all_infos, hnsw_index = future.result()
                index_list.append(hnsw_index)
                infos_list.append(all_infos)

        return index_list, infos_list

    def load_and_build_index(self, cupy_batch_files, infos_batch_files):
        all_infos = []
        cupy_embeddings = []

        for cupy_batch_file, infos_batch_file in zip(cupy_batch_files, infos_batch_files):
            batch_infos, batch_embeddings = self.load_infos_and_cupy_embeddings(cupy_batch_file, infos_batch_file)
            all_infos += batch_infos
            cupy_embeddings.append(batch_embeddings)

        cupy_embeddings = np.concatenate(cupy_embeddings, axis=0)
        index = cagra.build(cagra.IndexParams(build_algo='nn_descent'), cupy_embeddings)
        hnsw_index = hnsw.from_cagra(index)

        del cupy_embeddings
        del index
        cp.get_default_memory_pool().free_all_blocks()

        return all_infos, hnsw_index
        
    
    def search(self, query_path, top_k=20):
        total_neighbors = []
        total_distances = []
        query_image = self.request_image(query_path)

        query_embedding = self.image_encoder.encode_image(query_image)
        query_embedding = cp.asarray(query_embedding, dtype=cp.float32)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        #   query_embedding   NumPy  
        query_embedding = cp.asnumpy(query_embedding)


        # search_params = cagra.SearchParams()
        search_params = hnsw.SearchParams()
        result_infos = []

        for index,hnsw_index in tqdm(enumerate(self.index_list), desc="Searching",ncols=100):
            distances, neighbors = hnsw.search(search_params, hnsw_index, query_embedding, top_k)
            # distances, neighbors = brute_force.search(self.index, query_embedding, top_k)
            neighbors = cp.asarray(neighbors).flatten().tolist()
            distances = cp.asarray(distances).flatten().tolist()

            for neighbor in neighbors:
                result_infos.append(self.infos_list[index][neighbor])

            total_neighbors.extend(neighbors)
            total_distances.extend(distances)

    
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
            #  
            visited[node] = True
            #  
            component.append(search_info_list[node])
            
            #  
            for neighbor in range(len(search_info_list)):
                if not visited[neighbor] and self.judge_if_connected(search_info_list[node], search_info_list[neighbor]):
                    #  DFS
                    dfs(neighbor, component, visited)

        #  
        visited = [False] * len(search_info_list)
        components = []

        #  DFS 
        for i in range(len(search_info_list)):
            if not visited[i]:
                #  
                current_component = []
                #  DFS 
                dfs(i, current_component, visited)
                #  
                components.append(current_component)

        #  region component
        components = [component for component in components if len(component) > 1]

        return components

    def judge_if_connected(self, info1, info2):
        #  region , name level xy wh region xy 
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

#%%

if __name__ == "__main__":
    #%%
    begin = time.time()
    builder = Image2Image_Retriever_Rapid()
    end = time.time()
    print(f"Time cost: {end-begin}")
    print("Finish building the image vector database.")

    #%%
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



    print(f"Connected regions: {len(connected_regions)}")
    print("distances:", distances)
    print("neighbors:", neighbors)
    for neighbor in neighbors:
        print(builder.all_infos[neighbor])
        print("\n")
    
# %%
