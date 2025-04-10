import os, sys, json, qdrant_client, uuid
sys.path.append(".")
sys.path.append("..")
import aiohttp, asyncio
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from Retrieval_Server.modules.I2I_Retrieval.src.build_index_hpc.embedding import CustomWSIDataset, ImagePatchEncoder
from qdrant_client.http import models as rest
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from multiprocessing import Manager
from collections import deque
import asyncio
from aiofiles import os as async_os
from aiofiles.os import listdir

UUID_NAMESPACE = uuid.NAMESPACE_URL


class WSI_Image_Vector_DB_Builder():
    def __init__(self):
        self.image_client_name = "LBP_WSI_Image"
        self.embed_cache_path = os.path.join("Retrieval_Server/cache", "vector_database", "image2image_retrieval_big_500w_all")  # test("237208-12.tiff")
        self.wsi_image_encoder = None
        self.image_names = self.load_json_file("Retrieval_Server/cache/wsi_patch_image/loaded_wsis.json")

        loaded_embeddings_path = "Retrieval_Server/cache/patch_embeddings/loaded_wsis.json"
        if os.path.exists(loaded_embeddings_path):
            self.loaded_embeddings = self.load_json_file(loaded_embeddings_path)
        else:
            self.loaded_embeddings = []
            os.makedirs(os.path.dirname(loaded_embeddings_path), exist_ok=True)

    def init_image_vector_db_client(self, embed_cache_path):
        image_client = qdrant_client.QdrantClient(path=embed_cache_path)
        if not image_client.collection_exists(self.image_client_name):
            image_client.create_collection(
                collection_name=self.image_client_name,
                vectors_config=rest.VectorParams(size=1024, distance=rest.Distance.COSINE, datatype=rest.Datatype.FLOAT32),
                on_disk_payload=True,
                optimizers_config=rest.OptimizersConfigDiff(indexing_threshold=0),    
            )

        return image_client

    def load_json_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data
    
    def load_wsi_image_encoder(self):
        if self.wsi_image_encoder == None:
            self.wsi_image_encoder = ImagePatchEncoder()

    async def async_listdir(self, folder_path):

        return await listdir(folder_path)

    async def load_patch_infos(self, folder_path):

        patch_infos = await self.async_listdir(folder_path)
        return patch_infos

    async def load_image_paths(self, folder_path):

        image_paths = []
        filenames = await self.async_listdir(folder_path)
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(folder_path, filename))
        return image_paths

    async def loading_wsi_image(self, wsi_name):

        if wsi_name not in self.image_names:
            print(f"WSI {wsi_name} is not loaded.")
            return [], []

        folder_path = os.path.join("Retrieval_Server/cache/wsi_patch_image", wsi_name)
        patch_infos, image_paths = await asyncio.gather(
            self.load_patch_infos(folder_path),
            self.load_image_paths(folder_path)
        )

        wsi_dataset = CustomWSIDataset(image_paths)
        dataloader = DataLoader(wsi_dataset, batch_size=16, shuffle=False, num_workers=32, pin_memory=True)

        return patch_infos, dataloader

    def encoding_wsi_image(self, wsi_name, dataloader):

        patch_embeddings = self.wsi_image_encoder.get_wsi_patch_info_and_emb(wsi_name, dataloader)

        return patch_embeddings

    def insert_wsi_parload(self, wsi_name, patch_infos, patch_embeddings):

        image_client = self.init_image_vector_db_client(self.embed_cache_path)
        patch_ids, patch_payloads = [], []

        for info in tqdm(patch_infos, desc=f"WSI name: {wsi_name} for Info", ascii=True):
            patch_id = str(uuid.uuid5(UUID_NAMESPACE, wsi_name + info))
            patch_ids.append(patch_id)

            info_list = info[:-4].split("_")    # x_y_width_height_level
            payload = {
                    "position":(info_list[0],info_list[1]),
                    "patch_size":(info_list[2], info_list[3]),
                    "level":info_list[4],
                    "wsi_name":wsi_name,
                }
            
            patch_payloads.append(payload)

        # print(len(patch_ids), len(patch_payloads), len(patch_embeddings))

        if patch_ids and patch_payloads:
            response = image_client.upsert(
                collection_name=self.image_client_name,
                points=rest.Batch(
                    ids=patch_ids,
                    payloads=patch_payloads,
                    vectors=patch_embeddings,
                )
            )

    def loading_worker(self, input_queue, output_queue):
        while True:
            wsi_name = input_queue.get()
            if wsi_name is None:
                break

            if wsi_name in self.loaded_embeddings:
                print(f"WSI {wsi_name} cached.")
                output_queue.put((wsi_name, None, None))
            else:
                self.load_wsi_image_encoder()
                patch_infos, dataloader = asyncio.run(self.loading_wsi_image(wsi_name))
                if dataloader:
                    output_queue.put((wsi_name, patch_infos, dataloader))

    def encoding_worker(self, input_queue, output_queue):
        while True:
            item = input_queue.get()
            if item is None:
                break
            wsi_name, patch_infos, dataloader = item

            if wsi_name in self.loaded_embeddings:
                output_queue.put((wsi_name, None, None))
            else:
                patch_embeddings = self.encoding_wsi_image(wsi_name, dataloader)
                output_queue.put((wsi_name, patch_infos, patch_embeddings))

    def saving_worker(self, input_queue):
        while True:
            item = input_queue.get()
            if item is None:
                break
            wsi_name, patch_infos, patch_embeddings = item

            patch_infos_path = f"Retrieval_Server/cache/patch_embeddings/{wsi_name}/patch_infos.json"
            patch_embeddingss_path = f"Retrieval_Server/cache/patch_embeddings/{wsi_name}/patch_embeddings.json"

            if wsi_name in self.loaded_embeddings:
                with open(patch_infos_path, 'r') as load_patch_infos, open(patch_embeddingss_path, 'r') as load_patch_embeddings:
                    loaded_patch_infos = json.load(load_patch_infos)
                    loaded_patch_embeddings = json.load(load_patch_embeddings)
                    self.insert_wsi_parload(wsi_name, loaded_patch_infos, loaded_patch_embeddings)
            else:   
                self.insert_wsi_parload(wsi_name, patch_infos, patch_embeddings)
                os.makedirs(os.path.dirname(patch_infos_path), exist_ok=True)
                os.makedirs(os.path.dirname(patch_embeddingss_path), exist_ok=True)
                os.makedirs(os.path.dirname("Retrieval_Server/cache/patch_embeddings/loaded_wsis.json"), exist_ok=True)
                with open(patch_infos_path, 'w') as patch_infos_file, open(patch_embeddingss_path, 'w') as patch_embeddings_file:
                    json.dump(patch_infos, patch_infos_file, indent=4)
                    json.dump(patch_embeddings, patch_embeddings_file, indent=4)
                with open("Retrieval_Server/cache/patch_embeddings/loaded_wsis.json", 'w') as f:
                    self.loaded_embeddings.append(wsi_name)
                    json.dump(self.loaded_embeddings, f, indent=4)

    def process_wsis(self, wsi_names_list):
        load_workers = 16
        save_workers = 1

        manager = mp.Manager()
        load_queue = manager.Queue(maxsize=8)
        encode_queue = manager.Queue(maxsize=8)
        save_queue = manager.Queue(maxsize=8)

        loading_processes = [mp.Process(target=self.loading_worker, args=(load_queue, encode_queue)) 
                             for _ in range(load_workers)]
        encoding_process = mp.Process(target=self.encoding_worker, args=(encode_queue, save_queue))
        saving_processes = [mp.Process(target=self.saving_worker, args=(save_queue,))
                          for _ in range(save_workers)]

        for p in loading_processes:
            p.start()
        encoding_process.start()
        for p in saving_processes:
            p.start()


        for wsi_name in wsi_names_list:
            load_queue.put(wsi_name)

        for _ in range(load_workers):
            load_queue.put(None)
        for p in loading_processes:
            p.join()

        encode_queue.put(None)
        encoding_process.join()

        for _ in range(save_workers):
            save_queue.put(None)
        for p in saving_processes:
            p.join()

                        

if __name__ == "__main__":
    mp.set_start_method('spawn')

    wsi_name_path = "Retrieval_Server/modules/I2I_Retrieval/src/wsi_names.json"
    builder = WSI_Image_Vector_DB_Builder()
    wsi_name_list = builder.load_json_file(wsi_name_path)

    # wsi_name_list = [
    #     "237208-12.tiff",
    #     "241183-21.tiff",
    # ]
    builder.process_wsis(wsi_name_list)  