# import os, sys, json, qdrant_client, requests, cProfile, uuid, gc, logging
# sys.path.append(".")
# sys.path.append("..")
# sys.path.append("...")
# from io import BytesIO
# from PIL import Image
# from time import time
# from tqdm import tqdm
# from Retrieval_Server.modules.embeddings import Custom_UNI_Image_Embedding
# import asyncio
# import aiohttp
# from PIL import Image
# import cProfile
# import psutil
# from collections import deque
# from qdrant_client.http.models import PointStruct
# from qdrant_client.http import models as rest


# UUID_NAMESPACE = uuid.NAMESPACE_URL

# class Image2Image_vector_db():
#     def __init__(self):
#         self.image_client_name = "LBP_WSI_Image"
#         embed_cache_path = os.path.join("Retrieval_Server/caches", "vector_database", "image2image_retrieval_big")
#         self.image_client = self.init_image_vector_db_client(embed_cache_path)
#         self.image_embed_model = Custom_UNI_Image_Embedding()

#         self.image_load_limit = 5
#         self.loaded_images = deque([])
#         self.semaphore = asyncio.Semaphore(self.image_load_limit)   # max_concurrent_downloads
#         print("Finish Initialized Vector Databases.")
                    
#     async def asy_download_image(self, url):
#         async with self.semaphore:    
#             try:
#                 async with aiohttp.ClientSession() as session:
#                     async with session.get(url) as response:
#                         if response.status == 200:
#                             img_data = await response.read()
#                             try:
#                                 image = Image.open(BytesIO(img_data)).convert("RGB")
#                                 return image
#                             except Exception as e:
#                                 logging.error(f"Error opening image from {url}: {e}")
#                                 return None
#                         else:
#                             logging.error(f"Error downloading image from {url}: HTTP {response.status}")
#                             return None
#             except aiohttp.ClientError as e:
#                 logging.error(f"Network error occurred while downloading image from {url}: {e}")
#                 return None
#             except Exception as e:
#                 logging.error(f"Unexpected error occurred: {e}")
#                 return None
                    
#     def download_image(self, url):
#         with requests.get(url) as response:
#             if response.status_code == 200:
#                 img_data = response.content
#                 try:
#                     image = Image.open(BytesIO(img_data)).convert("RGB")
#                     return image
#                 except Exception as e:
#                     print(f"Error opening image from {url}: {e}")
#                     return None
#             else:
#                 print(f"Error downloading image from {url}: HTTP {response.status}")
#                 return None

#     def insert_wsi(self, wsi_name):
#         # NOTE - Example of url: "  /wsi/metaservice/api/region/openslide/237208-23.tiff/13312/20480/1024/1024/1"
#         wsi_url = f"  /wsi/metaservice/api/region/openslide/{wsi_name}"
#         wsi_info_url = wsi_url.replace("region", "sliceInfo")
#         print(wsi_info_url)
#         try:
#             slide_info = eval(requests.get(wsi_info_url, stream=True).content)
#         except:
#             print(f"CAn not find the information of {wsi_info_url}.")
#             return

#         slice_size = (128, 128)
#         num_level = int(slide_info["openslide.level-count"])

#         for level in range(num_level):
#             width = int(slide_info[f"openslide.level[{level}].width"])
#             height = int(slide_info[f"openslide.level[{level}].height"])

#             if width > 32768 or height > 32768:       # 暂时的限制条件 
#                 continue

#             urls = []
#             for y in range(0, height, slice_size[1]):
#                 urls.append(('/').join([wsi_url, "0", str(y), str(width), str(slice_size[1]), str(level)]))

#             column_range = tqdm(range(0, height, slice_size[1]))
#             for y in column_range:
#                 column_range.set_description(desc=f"Processing WSI: {wsi_name} in Level: {level} (Memory Usage: {psutil.virtual_memory().percent:.2f}%)")

#                 url = ('/').join([wsi_url, "0", str(y), str(width), str(slice_size[1]), str(level)])
#                 row_image = asyncio.run(self.asy_download_image(url))
#                 # row_image = self.check_downloaded_images(y // slice_size[1], urls)                  
#                 if row_image is None:
#                     continue                                                                     

#                 patch_img_list = []
#                 for x in range(0, width, slice_size[0]):
#                     patch_img_list.append(row_image.crop((x, 0, x+slice_size[0], y+slice_size[1]))) 
#                 del row_image

#                 embeddings = self.image_embed_model._get_image_embeddings(patch_img_list)         

#                 img_ids, image_payloads = [], []
#                 for x in range(0, width, slice_size[0]):
#                     patch_img_url = ('/').join([wsi_url, str(x), str(y), str(slice_size[0]), str(slice_size[1]), str(level)])

#                     payload = {
#                         "image_url":patch_img_url, 
#                         "position":(x,y),
#                         "level":level,
#                         "size":slice_size,
#                         "wsi_image_source":wsi_url,
#                     }

#                     img_ids.append(str(uuid.uuid5(UUID_NAMESPACE, patch_img_url)))
#                     image_payloads.append(payload)

#                 self.image_client.upload_collection(
#                     collection_name=self.image_client_name,
#                     vectors=embeddings,
#                     payload=image_payloads,
#                     ids=img_ids)

#     def init_image_vector_db_client(self, embed_cache_path):
#         image_client = qdrant_client.QdrantClient(path=embed_cache_path)
#         if not image_client.collection_exists(self.image_client_name):
#             image_client.create_collection(
#                 collection_name=self.image_client_name,
#                 vectors_config=rest.VectorParams(size=1024, distance=rest.Distance.COSINE),
#                 on_disk_payload=True,
#             )

#         return image_client
    
#     def retrieval(self, query_img_url, top_k=10):
#         query_img = self.download_image(query_img_url)
#         query_embedding = self.image_embed_model.get_image_embedding(query_img)

#         results = self.image_client.search(
#             collection_name="LBP_Image",
#             query=query_embedding,
#             with_payload=True,
#             limit=top_k,
#         ).points

#         return results

# def load_wsi_names():
#     filename = "Retrieval_Server/modules/I2I_Retrieval/src/wsi_names.json"
#     with open(filename, 'r', encoding='utf-8') as f:
#         data = json.load(f) 
#         return data
    

# if __name__ == "__main__":
#     wsi_vec_db = Image2Image_vector_db()

#     # wsi_name_list = load_wsi_names()
#     wsi_name_list = ["II20-11003#HE.svs"]

#     for wsi_name in wsi_name_list:
#         wsi_vec_db.insert_wsi(wsi_name)
    
#     query_img_url = "  /wsi/metaservice/api/region/openslide/237208-23.tiff/0/0/256/256/3"
#     results = wsi_vec_db.retrieval(query_img_url)

#     for point in results:
#         print(point.payload["image_url"])




#     ### NOTE: Statistical memory condition
#     # tracemalloc.start(25)

#     # try:
#     #     Image2Image_vector_db()
#     # except:
#     #     print("OOM")

#     # snapshot = tracemalloc.take_snapshot()
#     # top_stats = snapshot.statistics('traceback')

#     # stat = top_stats[0]
#     # print("%s memory blocks: %.1f KiB" % (stat.count, stat.size / 1024))
#     # for line in stat.traceback.format():
#     #     print(line)

#     ### NOTE: Profiling Process
#     # profile = cProfile.Profile()
#     # profile.enable()
#     # Image2Image_vector_db()
#     # profile.disable()

#     Image2Image_vector_db()

#     # s = StringIO()
#     # ps = pstats.Stats(profile, stream=s).sort_stats('tottime')
#     # ps.print_stats(50)
#     # print(s.getvalue())

#     # query_img_url = "  /wsi/metaservice/api/region/openslide/237208-23.tiff/0/0/256/256/3"
#     # vector_db = Image2Image_vector_db()
#     # retrieved_image_nodes = vector_db.retrieval(query_img_url)

#     # for node in retrieved_image_nodes:
#     #     print("Image_url", node.image_path)
#     #     print(f"Positions: {node.metadata["position"]}")
#     #     print(f"Level: {node.metadata["level"]}")
#     #     print(f"slice_size: {node.metadata["size"]}")
#     #     print(f"WSI_url: {node.metadata["wsi_image_source"]}")



