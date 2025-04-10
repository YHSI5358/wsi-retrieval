import numpy as np
import faiss
import os
import cupy as cp
import time
import json
import requests
from PIL import Image
from io import BytesIO
import queue
from threading import Thread

from MDI_RAG_Image2Image_Research.src.utils.encoder import WSI_Image_UNI_Encoder

def get_cpu_index_and_infos():
    cupy_dir = os.path.join(f"MDI_RAG_Image2Image_Research/data/cupy_embeddings_level1+")
    cupy_files = os.listdir(cupy_dir)
    cupy_files = [os.path.join(cupy_dir, file) for file in cupy_files]
    cupy_info_dir = os.path.join(f"MDI_RAG_Image2Image_Research/data/cupy_infos_level1+")

    cuda_type = "cupy_flatL2"

    cuda_cache_dir_name = cuda_type + "_embeddings_level1+"
    cuda_infos_dir_name = cuda_type + "_infos_level1+"
    cuda_index_dir_name = cuda_type + "_index_level1+"

    cuda_index_dir = "MDI_RAG_Image2Image_Research/data/" + cuda_index_dir_name
    cuda_infos_dir = "MDI_RAG_Image2Image_Research/data/" + cuda_infos_dir_name
    cuda_files = os.listdir(cuda_index_dir)

    index_list = []
    for i in range(len(cuda_files)):
        temp_index = faiss.read_index(f"{cuda_index_dir}/{cuda_type}_index_batch_{i}.bin")
        index_list.append(temp_index)

    info_list = []
    for i in range(len(cuda_files)):
        with open(f"{cuda_infos_dir}/{cuda_type}_infos_batch_{i}.json", "r") as f:
            info = json.load(f)
            info_list.append(info)
    return index_list, info_list

def request_image(query_img_path):
    if "http" in query_img_path:
        response = requests.get(query_img_path, verify=False)
        query_image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        query_image = Image.open(query_img_path).convert("RGB")
    return query_image

def save_result_info2img(result_infos):
    for info in result_infos:
        search_info = {}
        level = info.split("_")[-1].split(".")[0]
        w = info.split("_")[-3]
        h = info.split("_")[-2]
        x = info.split("_")[-5]
        y = info.split("_")[-4]
        name = "_".join(info.split("_")[:-5])

        new_url = f""
        query_img_path = new_url
        print(query_img_path)
        response = requests.get(query_img_path, verify=False)
        img = Image.open(BytesIO(response.content))
        img.save(f"test_img_faiss/{info}")

def produce_gpu_indices(index_list, gpu_queue):
    for i, index in enumerate(index_list):
        gpu_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
        gpu_queue.put((gpu_index, i))

def consume_gpu_indices(gpu_queue, query_image, info_list, result_infos):
    while True:
        gpu_index, index_id = gpu_queue.get()
        if gpu_index is None:
            break
        D, I = gpu_index.search(query_image, 10)
        for j in range(10):
            result_infos.append(info_list[index_id][I[0][j]])
        print(f"index_id: {index_id} finished")
        gpu_queue.task_done()

def main():
    image_encoder = WSI_Image_UNI_Encoder()
    query_img_path = ""
    query_image = request_image(query_img_path)
    query_image = image_encoder.encode_image(query_image)
    query_image = np.array(query_image).astype('float32')
    if query_image.ndim == 1:
        query_image = query_image.reshape(1, -1)

    index_list, info_list = get_cpu_index_and_infos()

    result_infos = []
    cost_time = 0
    gpu_queue = queue.Queue(maxsize=3)

    producer_thread = Thread(target=produce_gpu_indices, args=(index_list, gpu_queue))
    consumer_threads = [Thread(target=consume_gpu_indices, args=(gpu_queue, query_image, info_list, result_infos)) for _ in range(3)]

    time_start = time.time()
    producer_thread.start()
    for t in consumer_threads:
        t.start()

    producer_thread.join()
    for _ in range(3):
        gpu_queue.put((None, None))
    for t in consumer_threads:
        t.join()
    time_end = time.time()
    cost_time += time_end - time_start

    print(f"time:{cost_time}")
    save_result_info2img(result_infos)

if __name__ == "__main__":
    main()