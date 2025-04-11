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
# sys.path.append(os.path.join(os.path.dirname(__file__), '../../utils'))
# from encoder import WSI_Image_UNI_Encoder
from MDI_RAG_Image2Image_Research.src.utils.encoder import WSI_Image_UNI_Encoder
import time

import numpy as np
from cuvs.neighbors import hnsw
import concurrent.futures
import multiprocessing
import cProfile
import pstats
from flask import Flask, request, jsonify


app = Flask(__name__)


UUID_NAMESPACE = uuid.NAMESPACE_URL

#%%
# only cupy or faiss or ivf_flat
method_type = "cupy"

cuda_cache_dir_name = method_type + "_embeddings_level1+"
cuda_infos_dir_name = method_type + "_infos_level1+"
cuda_index_dir_name = method_type + "_index_level1+"

class Image2Image_Retriever_Rapid():
    def __init__(self):
        self.image_client_name = "WSI_Region_Retrieval"
        root_path = "/hpc2hdd/home/ysi538/retrieval/"
        self.embed_cache_path = f"/hpc2ssd/JH_DATA/spooler/ysi538/cupy_embeddings_batch_0.npy"
        self.cupy_cache_path = f"/hpc2ssd/JH_DATA/spooler/ysi538/cupy_embeddings_batch_0.npy"
        self.cupy_infos_path = f"/hpc2ssd/JH_DATA/spooler/ysi538/cupy_infos_batch_0.json"
        self.cupy_index_path = f"/hpc2ssd/JH_DATA/spooler/ysi538/cupy_index_batch_0.bin"
        self.image_encoder = WSI_Image_UNI_Encoder()
        self.index = self.load_index_file(self.cupy_index_path)
        self.infos = self.load_info_file(self.cupy_infos_path)

    def load_infos_and_cupy_embeddings(self, cupy_batch_files, infos_batch_files, begin=0, end=None):
        cupy_embeddings = []
        all_infos = []
        zip_list = list(zip(cupy_batch_files, infos_batch_files))

        for cupy_batch_file, infos_batch_file in zip_list[begin:end]:
            with open(os.path.join(self.cupy_cache_dir, cupy_batch_file), 'rb') as f:
                batch_embeddings = cp.load(f)
                #   cupy   numpy  
                batch_embeddings = cp.asnumpy(batch_embeddings)
                cupy_embeddings.append(batch_embeddings)
                #  
                del batch_embeddings
                cp.get_default_memory_pool().free_all_blocks()
            with open(os.path.join(self.cupy_infos_dir, infos_batch_file), 'r') as f:
                batch_infos = json.load(f)
                all_infos += batch_infos

        # cupy_embeddings = cp.concatenate(cupy_embeddings, axis=0)
        cupy_embeddings = np.concatenate(cupy_embeddings, axis=0)
        return all_infos,cupy_embeddings

    def load_index_file(self, index_file):
        index = hnsw.load(index_file, 1024, np.float32, "sqeuclidean")
        return index

    def load_info_file(self, info_file):
        with open(info_file, 'r') as f:
            infos = json.load(f)
        return infos


        
    
    def search(self, query_path, top_k=20):
        total_neighbors = []
        total_distances = []
        query_image = self.request_image(query_path)
        query_embedding = self.image_encoder.encode_image(query_image)
        #  numpy
        query_embedding = np.array(query_embedding).astype('float32')

        #   query_image  
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        result_infos = []

        if method_type == "cupy":
            search_params = hnsw.SearchParams()

            time_cost = 0
            hnsw_index = self.index
            begin_time = time.time()
            distances, neighbors = hnsw.search(search_params, hnsw_index, query_embedding, top_k)
            end_time = time.time()
            time_cost += end_time - begin_time
            # distances, neighbors = brute_force.search(self.index, query_embedding, top_k)
            neighbors = cp.asarray(neighbors).flatten().tolist()
            distances = cp.asarray(distances).flatten().tolist()
            for neighbor in neighbors:
                result_infos.append(self.infos[neighbor])
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
    
    def __del__(self):
        #  
        if hasattr(self, 'index_list'):
            for index in self.index_list:
                del index
        if cp is not None:
            cp.get_default_memory_pool().free_all_blocks()



image2image_retrieval = None

def get_image2image_retriever():
    """ """
    global image2image_retrieval
    if image2image_retrieval is None:
        print("============== image2image ==============")
        begin_time = datetime.now()
        image2image_retrieval = Image2Image_Retriever_Rapid()
        end_time = datetime.now()
        print("==============image2image =============="+" ", end_time - begin_time)
    return image2image_retrieval

@app.route('/test', methods=['POST'])
def process_image2image_retrieval():
    """ """
    query_img_path = request.json.get('query_img_path')
    top_k = request.json.get('top_k')

    image2image_retriever = get_image2image_retriever()
    distances, neighbors,results = image2image_retriever.search(query_img_path, top_k)
    # print(retrieved_images_payload)
    # neighbor "47062_EGFR- -230214010TFXA-0.7-LBP.ibl.tiff_5120_15872_256_256_1.png" 
    # results = [image2image_retriever.all_infos[neighbor] for neighbor in neighbors]


    retrieved_images_information = [{
            "wsi_level": node.split("_")[-1].split(".")[0],
            "metadata_position": (node.split("_")[-5], node.split("_")[-4]),
            "patch_size": (node.split("_")[-3], node.split("_")[-2]),
            "source_wsi_name":"_".join(node.split("_")[:-5]),
            # "wsi_id":node.split("_")[0],
        } for node in results]

    return jsonify({  
        'answer':"  Image2Image Retrieval  ",        #  
        'retrieved_images_information':retrieved_images_information
    })

if __name__ == "__main__":
    #%%
    get_image2image_retriever()
    app.run(debug=True, host='0.0.0.0', port="9876", use_reloader=False)
    # begin = time.time()
    # builder = Image2Image_Retriever_Rapid()
    # end = time.time()
    # print(f"Time cost: {end-begin}")
    # print("Finish building the image vector database.")

    # #%%
    # query_img_path =  " metaservice/api/region/openslide/241183-21.tiff/6400/25344/256/256/1"
    # # result= "6957_2200910-64_ .tiff_10496_4864_256_256_1.png"
    # # result_img_path = f" metaservice/api/region/openslide/2200910-64_ .tiff/10496/4864/256/256/1"
    # m=1
    # n=1
    # begin = time.time()
    # distances, neighbors, connected_regions,result_infos = builder.search_multi_imgs([query_img_path],m,n ,top_k=20)
    # end = time.time()
    # print(f"Time cost: {end-begin}")
    # for info in result_infos:
    #     print(info)
    #     print("\n")
    
# %%
# %%
