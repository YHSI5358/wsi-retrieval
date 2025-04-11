import os, sys, json, asyncio, aiohttp, requests, uuid, logging
sys.path.append(".")
sys.path.append("..")
sys.path.append("...")
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from datetime import datetime


class ImagePatchDownloader:
    def __init__(self, max_concurrent_downloads=100):
        self.url_head = " metaservice/api/region/openslide/"
        self.max_concurrent_downloads = max_concurrent_downloads
        self.slice_size = (256, 256)

        self.loaded_wsi_name_path = "Retrieval_Server/cache/wsi_patch_image/loaded_wsis.json"
        self.image_names = self.load_wsi_name(self.loaded_wsi_name_path)
    
    def load_wsi_name(self, json_file_path):

        if not os.path.exists(json_file_path):
            with open(json_file_path, "w") as file:
                json.dump([], file)
            return []

        with open(json_file_path, "r") as file:
            try:
                return json.load(file)
            except json.JSONDecodeError:
                return []

    def check_image_name(self, wsi_name):
        json_file_path = "Retrieval_Server/cache/wsi_patch_image/loaded_wsis.json"
        image_names = self.load_wsi_name(json_file_path)
        if wsi_name not in image_names:
            image_names.append(wsi_name)
            with open(json_file_path, "w") as file:
                json.dump(image_names, file, indent=4)
            return False
        else:
            return True

    def loading_wsi(self, wsi_name):

        if wsi_name in self.image_names:
            print(f"Patch of WSI {wsi_name} in the Cache.")
            return

        wsi_url = self.url_head + wsi_name
        wsi_info_url = wsi_url.replace("region", "sliceInfo")

        try:
            slide_info = eval(requests.get(wsi_info_url).content)
        except:
            print(f"Can not find the information of {wsi_info_url}.")
            return 

        if slide_info == {'error': 'No such file'}:
            
            print(wsi_info_url)
            return

        try:
            num_level = int(slide_info["openslide.level-count"])
        except:
            print(f"None useful num_level in wsi {wsi_name}")
            return

        patch_info_list = []
        for level in range(1, num_level):       
            width = int(slide_info[f"openslide.level[{level}].width"])
            height = int(slide_info[f"openslide.level[{level}].height"])

            for y in range(0, height, self.slice_size[1]):
                for x in range(0, width, self.slice_size[0]):
                    patch_infos = {
                        "x": str(x),
                        "y": str(y),
                        "width": str(self.slice_size[1]),
                        "height": str(self.slice_size[0]),
                        "level": str(level)
                    }
                    patch_info_list.append(patch_infos)
        
        wsi_dir_path = os.path.join(f"Retrieval_Server/cache/wsi_patch_image", wsi_name)
        os.makedirs(wsi_dir_path, exist_ok=True)
        asyncio.run(self.download_images(wsi_name, patch_info_list))

        self.image_names.append(wsi_name)
        with open(self.loaded_wsi_name_path, "w") as file:
            json.dump(self.image_names, file, indent=4)

    async def download_images(self, wsi_name, patch_infos):

        semaphore = asyncio.Semaphore(self.max_concurrent_downloads)
        async with semaphore:   
            
            async with aiohttp.ClientSession() as session:
                tasks = [self.asy_download_image(session, wsi_name, patch_info) for patch_info in patch_infos]
                with tqdm(total=len(tasks)) as pbar:
                    pbar.set_description(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | WSI: {wsi_name}")    
                    for task in asyncio.as_completed(tasks):
                        await task
                        pbar.update(1)
                        pbar.refresh() 


    async def asy_download_image(self, session, wsi_name, patch_info): 

        try:
            patch_url = os.path.join(self.url_head, wsi_name, ("/").join([patch_info[key] for key in patch_info]))
            async with session.get(patch_url) as response:
                if response.status == 200:
                    img_data = await response.read()
                    try:
                        image = Image.open(BytesIO(img_data)).convert("RGB")
                        patch_info = ("_").join([patch_info[key] for key in patch_info])
                        cache_path = os.path.join(f"Retrieval_Server/cache/wsi_patch_image", wsi_name, f"{patch_info}.png")
                        image.save(cache_path)
                    except Exception as e:
                        logging.error(f"Error opening image from {patch_url}: {e}")
                else:
                    logging.error(f"Error downloading image from {patch_url}: HTTP {response.status}")
        except aiohttp.ClientError as e:
            logging.error(f"Network error occurred while downloading image from {patch_url}: {e}")
        except Exception as e:
            logging.error(f"Unexpected error occurred: {e}")


if __name__ == "__main__":
    downloader = ImagePatchDownloader()
    file_path = "Retrieval_Server/modules/I2I_Retrieval/src/wsi_names.json"

    with open(file_path, 'r', encoding='utf-8') as f:
        wsi_name_list = json.load(f)
        for wsi_name in wsi_name_list:
            downloader.loading_wsi(wsi_name)