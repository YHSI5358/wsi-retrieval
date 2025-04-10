import os, sys, json, requests, qdrant_client, uuid
sys.path.append(".")
sys.path.append("..")
from qdrant_client.http import models as rest


### ------------------------------------------------------------------
### Using Embedding Servers to build WSI Patch Image Vector Database.
### Input: Json file of WSI names (json).
### Output: WSI Patch Image Vector Database (Qdrant).
### -----------------------------------------------------------------

UUID_NAMESPACE = uuid.NAMESPACE_URL

def wsi_image_encoding_response(wsi_name):
    url = "http://10.120.20.169:31990/wsi_image_embeddings"
    data = {'wsi_name':wsi_name,}

    response = requests.post(url, json=data)

    wsi_infos = response.json()['wsi_infos']
    embeddings = response.json()['embeddings']

    return wsi_infos, embeddings


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

    def process_wsis(self, wsi_name_list):
        slice_size = (256, 256)

        for wsi_name in wsi_name_list:
            wsi_infos_list, embeddings_list = wsi_image_encoding_response(wsi_name)
            for index in range(len(wsi_infos_list)):
                row_infos = wsi_infos_list[index]
                embeddings = embeddings_list[index]

                wsi_name, x, y, row_width, height, level = row_infos['name'], row_infos['x'], row_infos['y'], row_infos['width'], row_infos['height'], row_infos['level']
                wsi_url = f"  /wsi/metaservice/api/region/openslide/{wsi_name}"

                img_ids, image_payloads = [], []
                for x in range(0, row_width, slice_size[0]):
                    patch_img_url = ('/').join([wsi_url, str(x), str(y), str(slice_size[0]), str(slice_size[1]), str(level)])
                    payload = {
                        "image_url":patch_img_url, 
                        "position":(x,y),
                        "level":level,
                        "size":slice_size,
                        "wsi_image_source":wsi_name,
                    }
                    img_ids.append(str(uuid.uuid5(UUID_NAMESPACE, patch_img_url)))
                    image_payloads.append(payload)

                self.image_client.upload_collection(
                    collection_name=self.image_client_name,
                    vectors=embeddings,
                    payload=image_payloads,
                    ids=img_ids)

            


if __name__ == "__main__":
    file_path = "Retrieval_Server/modules/I2I_Retrieval/src/wsi_names.json"
    builder = WSI_Image_Vector_DB_Builder()
    wsi_name_list = builder.load_wsi_name_file(file_path)
    builder.process_wsis(wsi_name_list)