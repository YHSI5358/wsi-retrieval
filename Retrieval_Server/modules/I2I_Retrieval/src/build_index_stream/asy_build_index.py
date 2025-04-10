import os, sys, json, qdrant_client, requests, cProfile, uuid, gc, logging
sys.path.append(".")
sys.path.append("..")
sys.path.append("...")
from io import BytesIO
from PIL import Image
from time import time
from tqdm import tqdm
from Retrieval_Server.modules.embeddings import Custom_UNI_Image_Embedding
import asyncio
import aiohttp
from PIL import Image
import psutil
from collections import deque
from qdrant_client.http.models import PointStruct
from qdrant_client.http import models as rest
import concurrent.futures


UUID_NAMESPACE = uuid.NAMESPACE_URL

class ImageUrlDownloader:
    def __init__(self, max_concurrent_downloads=5):
        self.url_head = "  /wsi/metaservice/api/region/openslide/"
        self.max_concurrent_downloads = max_concurrent_downloads

    async def asy_download_image(self, session, url_info): 
        try:
            url = self.url_head + ("/").join([url_info[key] for key in url_info])
            async with session.get(url) as response:
                if response.status == 200:
                    img_data = await response.read()
                    try:
                        image = Image.open(BytesIO(img_data)).convert("RGB")
                        return (image, url_info)         
                    except Exception as e:
                        logging.error(f"Error opening image from {url}: {e}")
                        return None
                else:
                    logging.error(f"Error downloading image from {url}: HTTP {response.status}")
                    return None
        except aiohttp.ClientError as e:
            logging.error(f"Network error occurred while downloading image from {url}: {e}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error occurred: {e}")
            return None

    async def download_images(self, url_infos):
        semaphore = asyncio.Semaphore(self.max_concurrent_downloads)
        async with semaphore:   
            async with aiohttp.ClientSession() as session:
                tasks = [self.asy_download_image(session, url_info) for url_info in url_infos]
                return await asyncio.gather(*tasks)


class Image2Image_vector_db():
    def __init__(self):
        self.image_client_name = "LBP_WSI_Image"
        self.max_width = 8196
        self.max_patch_num = 256  
        embed_cache_path = os.path.join("Retrieval_Server/caches", "vector_database", "image2image_retrieval_big")
        self.image_client = self.init_image_vector_db_client(embed_cache_path)

        self.image_embed_model = Custom_UNI_Image_Embedding()
        self.image_downloader = ImageUrlDownloader(max_concurrent_downloads=20)
        print("Finish Initialized Vector Databases.")

    def load_wsi_name_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data
    
    def crop_image(self, row_image, x, slice_size):
        return row_image.crop((x, 0, x + slice_size[0], slice_size[1]))

    def parallel_cropper(self, row_image, width, slice_size):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            params = [(row_image.copy(), x, slice_size) for x in range(0, width, slice_size[0])]
            patch_img_list = list(executor.map(lambda p: self.crop_image(*p), params))
        return patch_img_list
        
    def insert_wsi(self, wsi_name):
        wsi_url = f"  /wsi/metaservice/api/region/openslide/{wsi_name}"
        wsi_info_url = wsi_url.replace("region", "sliceInfo")

        try:
            slide_info = eval(requests.get(wsi_info_url, stream=True).content)
        except:
            print(f"CAn not find the information of {wsi_info_url}.")
            return

        slice_size = (256, 256)
        num_level = int(slide_info["openslide.level-count"])

        wsi_row_url_infos = []    
        for level in range(1, num_level):     
            width = int(slide_info[f"openslide.level[{level}].width"])
            height = int(slide_info[f"openslide.level[{level}].height"])

            redundant = (width - 1) % self.max_width + 1
            for y in range(0, height, slice_size[1]):
                for x in range(0, width, self.max_width):
                    wsi_row_url_infos.append(
                        {"name":wsi_name, 
                         "x":str(x), 
                         "y":str(y), 
                         "width":str(min(self.max_width, redundant)), 
                         "height":str(slice_size[1]), 
                         "level":str(level)}
                    )
    
        for index in range(0, len(wsi_row_url_infos), self.max_patch_num):
            cur_infos = wsi_row_url_infos[index:index+self.max_patch_num]
            asyncio.run(self.insert_wsi_with_patch_info(cur_infos, wsi_name, slice_size))

    async def insert_wsi_with_patch_info(self, cur_infos, wsi_name, slice_size):
        wsi_url = f"  /wsi/metaservice/api/region/openslide/{wsi_name}"
        row_image_infos = await self.image_downloader.download_images(cur_infos)

        for row_image_info in tqdm(row_image_infos):
            print(row_image_info)
            if row_image_info is None:
                continue         
            row_image, row_infos = row_image_info                                                         
            y, patch_width, level = row_infos['y'], int(row_infos['width']), row_infos['level']

            patch_img_list = self.parallel_cropper(row_image, patch_width, slice_size)     
            del row_image

            embeddings = self.image_embed_model._get_image_embeddings(patch_img_list)  
            del patch_img_list

            img_ids, image_payloads = [], []
            for x in range(0, patch_width, slice_size[0]):
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
            del embeddings, img_ids, image_payloads

    def init_image_vector_db_client(self, embed_cache_path):
        image_client = qdrant_client.QdrantClient(path=embed_cache_path)
        if not image_client.collection_exists(self.image_client_name):
            image_client.create_collection(
                collection_name=self.image_client_name,
                vectors_config=rest.VectorParams(size=1024, distance=rest.Distance.COSINE),
                on_disk_payload=True,
            )

        return image_client
    

if __name__ == "__main__":
    file_path = "Retrieval_Server/modules/I2I_Retrieval/src/wsi_names.json"

    wsi_vec_db = Image2Image_vector_db()
    wsi_name_list = wsi_vec_db.load_wsi_name_file(file_path)
    # wsi_name_list = ["II20-11003#HE.svs"]

    wsi_name_list_range = tqdm(wsi_name_list)
    for wsi_name in wsi_name_list:
        wsi_name_list_range.set_description(desc=f"Processing WSI: {wsi_name} batch (Memory Usage: {psutil.virtual_memory().percent:.2f}%)")
        wsi_vec_db.insert_wsi(wsi_name)
    

    ### NOTE: Profiling Process
    # profile = cProfile.Profile()
    # profile.enable()
    # Image2Image_vector_db()
    # profile.disable()

    # Image2Image_vector_db()

    # s = StringIO()
    # ps = pstats.Stats(profile, stream=s).sort_stats('tottime')
    # ps.print_stats(50)
    # print(s.getvalue())

    # query_img_url = "  /wsi/metaservice/api/region/openslide/237208-23.tiff/0/0/256/256/3"
    # vector_db = Image2Image_vector_db()
    # retrieved_image_nodes = vector_db.retrieval(query_img_url)

    # for node in retrieved_image_nodes:
    #     print("Image_url", node.image_path)
    #     print(f"Positions: {node.metadata["position"]}")
    #     print(f"Level: {node.metadata["level"]}")
    #     print(f"slice_size: {node.metadata["size"]}")
    #     print(f"WSI_url: {node.metadata["wsi_image_source"]}")