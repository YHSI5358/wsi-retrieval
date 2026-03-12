import os, sys, json, requests, qdrant_client, uuid
sys.path.append(".")
sys.path.append("..")
import aiohttp, asyncio
from tqdm import tqdm
from qdrant_client.http import models as rest



UUID_NAMESPACE = uuid.NAMESPACE_URL


class WSI_Image_Vector_DB_Builder():
    def __init__(self):
        self.image_client_name = "LBP_WSI_Image"
        embed_cache_path = os.path.join("Retrieval_Server/caches", "vector_database", "image2image_retrieval_big")
        self.image_client = self.init_image_vector_db_client(embed_cache_path)

    def init_image_vector_db_client(self, embed_cache_path):
        image_client = qdrant_client.QdrantClient(path=embed_cache_path)
        if not image_client.collection_exists(self.image_client_name):
            image_client.create_collection(
                collection_name=self.image_client_name,
                vectors_config=rest.VectorParams(size=1024, distance=rest.Distance.COSINE),
                on_disk_payload=True,
            )

        return image_client
    
    def load_wsi_name_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data

    async def wsi_image_encoding_response(self, session, wsi_name):
        url = "http://10.120.20.169:32407/wsi_image_embeddings"
        data = {'wsi_name':wsi_name,}
        
        timeout = aiohttp.ClientTimeout(total=3600)  #  60 
        async with session.post(url, json=data, timeout=timeout) as response:
            response_json = await response.json()
            patch_infos = response_json['patch_infos']
            patch_embeddings = response_json['patch_embeddings']

        return patch_infos, patch_embeddings


    async def insert_parload(self, wsi_name, patch_infos, patch_embeddings):
        patch_ids, patch_payloads = [], []
        for info in patch_infos:
            patch_id = str(uuid.uuid5(UUID_NAMESPACE, info))
            patch_ids.append(patch_id)

            info_list = info[:-4].split("_")    # x_y_width_height_level
            payload = {
                    "position":(info_list[0],info_list[1]),
                    "patch_size":(info_list[2], info_list[3]),
                    "level":info_list[4],
                    "wsi_name":wsi_name,
                }
            patch_payloads.append(payload)

        if patch_ids and patch_payloads:
            self.image_client.upload_collection(
                collection_name=self.image_client_name,
                vectors=patch_embeddings,
                payload=patch_payloads,
                ids=patch_ids)

    async def process_wsis(self, wsi_name_list):
        async with aiohttp.ClientSession() as session:
            for wsi_name in tqdm(wsi_name_list):
                patch_infos, patch_embeddings = await self.wsi_image_encoding_response(session, wsi_name)
                await self.insert_parload(wsi_name, patch_infos, patch_embeddings)

if __name__ == "__main__":
    wsi_name_path = "Retrieval_Server/modules/I2I_Retrieval/src/wsi_names.json"
    builder = WSI_Image_Vector_DB_Builder()
    wsi_name_list = builder.load_wsi_name_file(wsi_name_path)

    # wsi_name_list = [
    #     "241183-21.tiff",
    #     "258992-29.tiff",
    #     "270493-18.tiff",
    # ]
    asyncio.run(builder.process_wsis(wsi_name_list))