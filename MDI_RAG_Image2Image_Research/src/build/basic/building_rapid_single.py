import os
import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("...")
import json
import uuid
import cupy as cp
from cuvs.neighbors import cagra, hnsw, brute_force, ivf_flat
from tqdm import tqdm
from datetime import datetime
import time

import faiss  
import numpy as np

UUID_NAMESPACE = uuid.NAMESPACE_URL

# only cupy or faiss or faiss_flatL2 or cupy_flatL2
method_type = "faiss_IVFFlat"

cuda_cache_dir_name = method_type + "_embeddings_level1+"
cuda_infos_dir_name = method_type + "_infos_level1+"
cuda_index_dir_name = method_type + "_index_level1+"


class WSI_Image_Vector_DB_Builder:
    def __init__(self):
        self.image_client_name = "WSI_Region_Retrieval"
        self.embed_cache_path = "MDI_RAG_Image2Image_Research/data/embedding_cache"

        self.cuda_cache_dir = "MDI_RAG_Image2Image_Research/data/" + cuda_cache_dir_name
        self.cuda_infos_dir = "MDI_RAG_Image2Image_Research/data/" + cuda_infos_dir_name
        self.cuda_index_dir = "MDI_RAG_Image2Image_Research/data/" + cuda_index_dir_name

        # self.cuda_cache_dir = "MDI_RAG_Image2Image_Research/data/faiss_embeddings_level1+"
        # self.cuda_infos_dir = "MDI_RAG_Image2Image_Research/data/faiss_infos_level1+"
        # self.cuda_index_dir = "MDI_RAG_Image2Image_Research/data/faiss_index_level1+"

        self.image_names = os.listdir(self.embed_cache_path)
        


    def build_all_infos_and_cuda_embeddings(self):
        if not os.path.exists(self.cuda_infos_dir):
            os.makedirs(self.cuda_infos_dir)

        if not os.path.exists(self.cuda_cache_dir):
            os.makedirs(self.cuda_cache_dir)

        if not os.path.exists(self.cuda_index_dir):
            os.makedirs(self.cuda_index_dir)

  
        print("Building all infos.")
        batch_size = 100
        num_batches = (len(self.image_names) + batch_size - 1) // batch_size

        for batch_index in range(num_batches):
            if os.path.exists(os.path.join(self.cuda_infos_dir, f"{method_type}_infos_batch_{batch_index}.json")):
                continue
            if os.path.exists(os.path.join(self.cuda_cache_dir, f"{method_type}_embeddings_batch_{batch_index}.npy")):
                continue
            temp_dir = os.path.join(f"MDI_RAG_Image2Image_Research/data/cupy_embeddings_level1+/", f"cupy_embeddings_batch_{batch_index}.npy")
            print(f"Reading cupy embeddings from {temp_dir}.")
            if os.path.exists(temp_dir):
                embeddings = cp.load(os.path.join(temp_dir))
                # 转换为numpy
                embeddings = cp.asnumpy(embeddings).astype(np.float32)
            else:
                print(f"Embeddings file not found for batch {batch_index}.")
                continue

            if method_type == "faiss_IVFFlat":

                # print(f"Reading cupy embeddings from {self.cuda_cache_dir}.")
                cupy_dir = os.path.join(f"MDI_RAG_Image2Image_Research/data/cupy_embeddings_level1+")
                               

                n = 2 

                res = faiss.StandardGpuResources()
                config = faiss.GpuIndexIVFFlatConfig()
                config.use_raft = False
                nlist = 1024

                embeddings = np.array(embeddings).astype('float32')
                print(f"Batch embeddings shape: {embeddings.shape}")
                print(f"type of embeddings: {type(embeddings)}")

                print(f"Training faiss index for batch {batch_index}.")
                count = embeddings.shape[0]
                all_infos = json.load(open(f"MDI_RAG_Image2Image_Research/data/cupy_infos_level1+/cupy_infos_batch_{batch_index}.json"))

  
                chunk_size = count // n

                for i in range(n):
                    start_idx = i * chunk_size
                    end_idx = (i + 1) * chunk_size if i < n - 1 else count  #
                    cuda_index = faiss.GpuIndexIVFFlat(res, embeddings.shape[1], nlist, faiss.METRIC_L2, config)
                    cuda_index.train(embeddings[start_idx:end_idx])
                    cuda_index.add(embeddings[start_idx:end_idx])
                    idx_cpu = faiss.index_gpu_to_cpu(cuda_index)
                    print(f"Saving faiss index to {self.cuda_index_dir}_{i}.")

                    batch_infos = all_infos[start_idx:end_idx] 
                    if n == 1:
                        self.save_index_to_file(idx_cpu, f"{batch_index}")
                        self.save_infos_to_file(batch_infos, f"{batch_index}")
                    else:
                        self.save_index_to_file(idx_cpu, f"{batch_index}_{i}")
                        self.save_infos_to_file(batch_infos, f"{batch_index}_{i}")


            if method_type == "faiss_flatL2":
                res = faiss.StandardGpuResources()
                flat_config = faiss.GpuIndexFlatConfig()
                flat_config.device = 0

                cupy_dir = os.path.join(f"MDI_RAG_Image2Image_Research/data/cupy_embeddings_level1+")
                temp_dir = os.path.join(f"MDI_RAG_Image2Image_Research/data/cupy_embeddings_level1+/", f"cupy_embeddings_batch_{batch_index}.npy")
                if os.path.exists(temp_dir):

                    print(f"Reading cupy embeddings from {temp_dir}.")
                    embeddings = cp.load(os.path.join(temp_dir))

                    embeddings = cp.asnumpy(embeddings).astype(np.float32)

               
                    print(f"Batch embeddings shape: {embeddings.shape}")
                    cuda_index = faiss.GpuIndexFlatL2(res, embeddings.shape[1], flat_config)
                    # cuda_index = faiss.IndexHNSWFlat(batch_embeddings.shape[1], 32)
                    cuda_index.add(embeddings)
                    idx_cpu = faiss.index_gpu_to_cpu(cuda_index)
                    all_infos = json.load(open(f"MDI_RAG_Image2Image_Research/data/cupy_infos_level1+/cupy_infos_batch_{batch_index}.json"))


                    self.save_index_to_file(idx_cpu, batch_index)
                    self.save_infos_to_file(all_infos, batch_index)

            if method_type == "cupy_flatL2":
                embeddings = cp.array(embeddings, dtype=cp.float32)
                print(f"Batch embeddings shape: {embeddings.shape}")
                cuda_index = brute_force.build(embeddings,metric='sqeuclidean')

                self.save_index_to_file(cuda_index, batch_index)
                # self.save_embeddings_to_file(embeddings, batch_index)
                self.save_infos_to_file(batch_infos, batch_index)

            




                    
            # if os.path.exists(os.path.join(self.cuda_index_dir, f"{method_type}_index_batch_{batch_index}.bin")):
            #     continue

            # start_index = batch_index * batch_size
            # end_index = min((batch_index + 1) * batch_size, len(self.image_names))
            # batch_image_names = self.image_names[start_index:end_index]

            # batch_infos = []
            # batch_embeddings = []
            # for wsi_name in tqdm(batch_image_names, desc=f"Batch index: {batch_index}", ascii=True, ncols=150):
            #     patch_info_path = os.path.join(self.embed_cache_path, wsi_name, "patch_info_edited.json")
            #     each_wsi_infos = self.load_json_file(patch_info_path)
                
            #     each_wsi_embeddings = self.load_embeddings(wsi_name)

      
            #     ignore_index = []

            #     each_wsi_infos = [info for i, info in enumerate(each_wsi_infos) if i not in ignore_index]
            #     each_wsi_embeddings = [embedding for i, embedding in enumerate(each_wsi_embeddings) if i not in ignore_index]

            #     batch_infos += each_wsi_infos
            #     batch_embeddings += each_wsi_embeddings

            

            # elif method_type == "cupy":

            #     batch_embeddings = cp.array(batch_embeddings, dtype=cp.float32)
            #     print(f"Batch embeddings shape: {batch_embeddings.shape}")
            #     cuda_index = cagra.build(cagra.IndexParams(build_algo = 'nn_descent'), batch_embeddings)

            #     self.save_index_to_file(cuda_index, batch_index)
            #     self.save_embeddings_to_file(batch_embeddings, batch_index)
            #     self.save_infos_to_file(batch_infos, batch_index)

            # elif method_type == "ivf_flat":

            #     batch_embeddings = np.array(batch_embeddings).astype(np.float32)
            #     print(f"Batch embeddings shape: {batch_embeddings.shape}")
            #     build_params = ivf_flat.IndexParams(metric="sqeuclidean")
            #     index = ivf_flat.build(build_params, batch_embeddings)


            #     self.save_index_to_file(index, batch_index)
            #     self.save_embeddings_to_file(batch_embeddings, batch_index)
            #     self.save_infos_to_file(batch_infos, batch_index)

            # 释放显存
            del cuda_index
            del embeddings
            cp.get_default_memory_pool().free_all_blocks()
            
        print("All infos built.")

    
    def save_infos_to_file(self, infos, batch_index):
        """将 infos 保存到文件。"""
        file_path = os.path.join(self.cuda_infos_dir, f"{method_type}_infos_batch_{batch_index}.json")
        with open(file_path, "w") as f:
            json.dump(infos, f)
    
    def save_embeddings_to_file(self, embeddings, batch_index):
        """将 cuda 数组保存到文件。"""
        if method_type == "faiss":
            embeddings = embeddings.astype(np.float32)
            file_path = os.path.join(self.cuda_cache_dir, f"{method_type}_embeddings_batch_{batch_index}.npy")
            np.save(file_path, embeddings)
        elif method_type == "cupy":
            file_path = os.path.join(self.cuda_cache_dir, f"{method_type}_embeddings_batch_{batch_index}.npy")
            cp.save(file_path, embeddings)

    def save_index_to_file(self, index, batch_index):
        """将 index 保存到文件。"""
        if method_type == "faiss":
            file_path = os.path.join(self.cuda_index_dir, f"{method_type}_index_batch_{batch_index}.bin")
            faiss.write_index(index, file_path)
        if method_type == "cupy":
            file_path = os.path.join(self.cuda_index_dir, f"{method_type}_index_batch_{batch_index}.bin")
            hnsw.save(file_path, index)
        if method_type == "faiss_flatL2":
            file_path = os.path.join(self.cuda_index_dir, f"{method_type}_index_batch_{batch_index}.bin")
            faiss.write_index(index, file_path)

        if method_type == "faiss_IVFFlat":
            file_path = os.path.join(self.cuda_index_dir, f"{method_type}_index_batch_{batch_index}.bin")
            faiss.write_index(index, file_path)
        # if method_type == "cupy_flatL2":
        #     file_path = os.path.join(self.cuda_index_dir, f"{method_type}_index_batch_{batch_index}.bin")
        #     hnsw.save(file_path, index)



    def load_embeddings(self, wsi_name):
        embeddings_path = os.path.join(self.embed_cache_path, wsi_name, "embeddings.json")
        embeddings = self.load_json_file(embeddings_path)
        return embeddings


    def load_json_file(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
            return data
        
    def update_image_names(self):
        self.image_names = os.listdir(self.embed_cache_path)
        
    def update_cuda_index(self):
        """ 对现有的 embeddings，如果没有索引，就建立索引。"""
        existing_index_files = sorted([f for f in os.listdir(self.cuda_index_dir) if f.startswith(f"{method_type}_index_batch_")])
        existing_index_files = [f.split("_")[-1].split(".")[0] for f in existing_index_files]

        existing_embeddings_files = sorted([f for f in os.listdir(self.cuda_cache_dir) if f.startswith(f"{method_type}_embeddings_batch_")])
        existing_embeddings_files = [f.split("_")[-1].split(".")[0] for f in existing_embeddings_files]
        print("Existing index files:", existing_index_files)
        print("Existing embeddings files:", existing_embeddings_files)

        for index_file in tqdm(existing_embeddings_files, desc="Building index", ascii=True, ncols=100):
            if index_file not in existing_index_files:
                print(f"Building index for batch {index_file}...")
                embeddings = cp.load(os.path.join(self.cuda_cache_dir, f"{method_type}_embeddings_batch_{index_file}.npy"))
                cuda_index = cagra.build(cagra.IndexParams(build_algo='nn_descent'), embeddings)
                self.save_index_to_file(cuda_index, index_file)
                del cuda_index
                del embeddings
                cp.get_default_memory_pool().free_all_blocks()
                print(f"Index for batch {index_file} built.")
    
    def update_new_wsi(self, remain_force = False):
        """更新 embeddings，处理新来的 WSI。"""
        self.update_image_names()

        existing_wsi_names = set()
        for file_name in os.listdir(self.cuda_infos_dir):
            if file_name.endswith(".json"):
                infos = self.load_json_file(os.path.join(self.cuda_infos_dir, file_name))
                # 18632_L02_CD4.svs_0_0_256_256_1.png
                # existing_wsi_names.update({"_".join(info.split("_")[1:-5]) for info in infos})
                existing_wsi_names.update({"_".join(info.split("_")[:-5]) for info in infos})

        new_wsi_names = [name for name in self.image_names if name not in existing_wsi_names]

        if not new_wsi_names:
            print("No new WSI found.")
            return

        print(f"Found {len(new_wsi_names)} new WSI. Updating embeddings...")

        batch_size = 100
        num_batches = (len(new_wsi_names) + batch_size - 1) // batch_size

        if len(new_wsi_names) < batch_size and not remain_force:
            print("Batch size is larger than the number of new WSI. Stop update.")
            return

        for batch_index in range(num_batches):
            start_index = batch_index * batch_size
            end_index = min((batch_index + 1) * batch_size, len(new_wsi_names))
            batch_image_names = new_wsi_names[start_index:end_index]
            if len(batch_image_names) < batch_size and not remain_force:
                print("Batch size is smaller than the number of new WSI. Stop update.")
                break
                

            embeddings = [self.load_embeddings(wsi_name) for wsi_name in tqdm(batch_image_names, desc=f"Batch index: {batch_index}", ascii=True, ncols=150)]
            embeddings = [item for sublist in embeddings for item in sublist]  # Flatten the list
            embeddings = cp.array(embeddings, dtype=cp.float32)
            

            # 保存新的 embeddings
            existing_batch_files = sorted([f for f in os.listdir(self.cuda_cache_dir) if f.startswith(f"{method_type}_embeddings_batch_")])
            new_batch_index = len(existing_batch_files)
            

            
            # 更新 infos
            batch_infos = []
            for wsi_name in batch_image_names:
                patch_info_path = os.path.join(self.embed_cache_path, wsi_name, "patch_info_edited.json")
                if not os.path.exists(patch_info_path):
                    patch_info_path_edited = os.path.join(self.embed_cache_path, wsi_name, 'patch_info_edited.json')
                    patch_info_path_origin = os.path.join(self.embed_cache_path, wsi_name, 'patch_info.json')
                    with open(patch_info_path_origin, 'r') as f:
                        patch_info_origin = json.load(f)
                        # 在最前面加上图片名
                        patch_info_edited = [wsi_name + i for i in patch_info_origin]       
                    # 将修改后的patch_info写回文件
                    with open(patch_info_path_edited, 'w') as f:
                        json.dump(patch_info_edited, f)

                patch_infos = self.load_json_file(patch_info_path)
                batch_infos += patch_infos



            # 更新 index
            existing_index_files = sorted([f for f in os.listdir(self.cuda_index_dir) if f.startswith(f"{method_type}_index_batch_")])
            existing_index_files = [f.split("_")[-1].split(".")[0] for f in existing_index_files]
            if new_batch_index not in existing_index_files:
                cuda_index = cagra.build(cagra.IndexParams(build_algo='nn_descent'), embeddings)
                self.save_index_to_file(cuda_index, new_batch_index)
                del cuda_index

            
            self.save_embeddings_to_file(embeddings, new_batch_index)
            self.save_infos_to_file(batch_infos, new_batch_index)
                
            # 释放显存
            del embeddings
            cp.get_default_memory_pool().free_all_blocks()

        print("Embeddings and infos updated.")



if __name__ == "__main__":
    builder = WSI_Image_Vector_DB_Builder()
    builder.build_all_infos_and_cuda_embeddings()
    # 每10分钟更新一次


    # while True:
    #     print("=====================================================")
    #     print("Time:", datetime.now())
    #     # builder.update_new_wsi(remain_force = True)
    #     # builder.build_all_infos_and_cuda_embeddings()
    #     time.sleep(3600)



    # builder.update_cuda_index()
    # print("Finish building the image vector database.")