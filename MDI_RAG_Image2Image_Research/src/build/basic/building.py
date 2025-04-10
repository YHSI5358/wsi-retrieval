import os, sys, json, qdrant_client, uuid
sys.path.append(".")
sys.path.append("..")
from tqdm import tqdm
from datetime import datetime
from qdrant_client.http import models as rest
import multiprocessing as mp


UUID_NAMESPACE = uuid.NAMESPACE_URL


class WSI_Image_Vector_DB_Builder():
    def __init__(self):
        self.image_client_name = "WSI_Region_Retrieval"
        self.embed_cache_path = "MDI_RAG_Image2Image_Research/data/embedding_cache"
        self.database_path = "MDI_RAG_Image2Image_Research/data/vector_database"
        self.image_names = os.listdir(self.embed_cache_path)

    def load_json_file(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
            return data

    def init_image_vector_db_client(self, database_path):
        image_client = qdrant_client.QdrantClient(path=database_path)
        if not image_client.collection_exists(self.image_client_name):
            image_client.create_collection(
                collection_name=self.image_client_name,
                vectors_config=rest.VectorParams(size=1024, distance=rest.Distance.COSINE, datatype=rest.Datatype.FLOAT32),
                on_disk_payload=True,
            )

        return image_client
    
    def loading_worker(self, input_queue, output_queue):
        while True:
            wsi_name = input_queue.get()
            if wsi_name is None:
                break

            patch_info_path = os.path.join(self.embed_cache_path, wsi_name, "patch_info.json")
            patch_infos = self.load_json_file(patch_info_path)
            embeddings_path = os.path.join(self.embed_cache_path, wsi_name, "embeddings.json")
            embeddings = self.load_json_file(embeddings_path)
            
            output_queue.put((wsi_name, patch_infos, embeddings))

    def insert_worker(self, input_queue):
        while True:
            item = input_queue.get()
            if item is None:
                break
            wsi_name, patch_infos, patch_embeddings = item
            print(f"Inserting {wsi_name} into the database.")
            print(f"Number of patches: {len(patch_infos)}")
            print(f"Number of embeddings: {len(patch_embeddings)}")

            image_client = self.init_image_vector_db_client(self.database_path)
            patch_ids, patch_payloads = [], []
            patch_infos_pbar = tqdm(patch_infos, ascii=True)
            for info in patch_infos_pbar:
                current_time = datetime.now().strftime("%H:%M:%S")
                patch_infos_pbar.set_description(f"WSI name: {wsi_name}, Time: {current_time}")

                patch_id = str(uuid.uuid5(UUID_NAMESPACE, wsi_name + info))
                patch_ids.append(patch_id)

                info_list = info[1:-4].split("_")
                payload = {
                        "position":(info_list[0],info_list[1]),
                        "patch_size":(info_list[2], info_list[3]),
                        "level":info_list[4],
                        "wsi_name":wsi_name,
                    }
                
                patch_payloads.append(payload)

            image_client.upsert(
                collection_name=self.image_client_name,
                points=rest.Batch(
                    ids=patch_ids,
                    payloads=patch_payloads,
                    vectors=patch_embeddings,
                )
            )
            image_client.close()

    def main(self, wsi_names_list):
        load_workers = 4
        load_queue = mp.Queue(maxsize=8)
        save_queue = mp.Queue()

        loading_processes = [mp.Process(target=self.loading_worker, args=(load_queue, save_queue)) 
                             for _ in range(load_workers)]
        saving_process = mp.Process(target=self.insert_worker, args=(save_queue,))

        for p in loading_processes:
            p.start()
        saving_process.start()

        for wsi_name in wsi_names_list:
            load_queue.put(wsi_name)

        for _ in range(load_workers):
            load_queue.put(None)
        for p in loading_processes:
            p.join()
        save_queue.put(None)
        saving_process.join()

                        

if __name__ == "__main__":
    mp.set_start_method('spawn')
    builder = WSI_Image_Vector_DB_Builder()
    wsi_names_list = os.listdir(builder.embed_cache_path)
    builder.main(wsi_names_list)  