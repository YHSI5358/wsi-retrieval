import os, sys, json, asyncio, aiohttp, requests, uuid, logging, timm
sys.path.append(".")
sys.path.append("..")
sys.path.append("...")
from io import BytesIO
from PIL import Image
from tqdm import tqdm
import concurrent.futures
from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor


### -------------------------------------------------------
### This Python file using for Embedding Servers.
### Input: Name of WSI (str).
### Output: Patch image Embeddings (List).
### -------------------------------------------------------


UUID_NAMESPACE = uuid.NAMESPACE_URL

class ImageUrlDownloader:
    def __init__(self, max_concurrent_downloads=5):
        self.url_head = " metaservice/api/region/openslide/"
        # self.url_head = "https://10.108.5.177/wsi/metaservice/api/region/openslide/"
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
            # timeout = aiohttp.ClientTimeout(total=60)  
            async with aiohttp.ClientSession() as session:
                tasks = [self.asy_download_image(session, url_info) for url_info in url_infos]
                return await asyncio.gather(*tasks)

class WSI_Image_UNI_Encoder():
    def __init__(self, *, embed_batch_size=1024, **kwargs):
        self.embed_model =  timm.create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )
        self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])

        local_dir = "checkpoints/vit_large_patch16_224.dinov2.uni_mass100k/"
        self._device = self.infer_torch_device()
        print(self._device)
        self.embed_model = self.embed_model.to(self._device)
        self.embed_model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
        self.embed_model.eval()

    def infer_torch_device(self):
        """Infer the input to torch.device."""
        try:
            has_cuda = torch.cuda.is_available()
        except NameError:
            import torch  # pants: no-infer-dep
            has_cuda = torch.cuda.is_available()
        if has_cuda:
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def transform_image(self, image, transform):
        return transform(image)

    def parallel_transform(self, images, transform, device):
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.transform_image, image, transform) for image in images]
            transformed_images = [future.result() for future in futures]

        return torch.stack(transformed_images, dim=0).to(device)

    def _get_image_list_embeddings(self, img_list):
        images_tensor = self.parallel_transform(img_list, self.transform, self._device)
        with torch.inference_mode():
            feature_emb_tensor = self.embed_model(images_tensor) 
        
        return feature_emb_tensor.cpu().tolist()


class WSI_Image_Encoder_server():
    def __init__(self):
        self.max_width = 256 * 128
        self.image_embed_model = WSI_Image_UNI_Encoder()
        self.image_downloader = ImageUrlDownloader(max_concurrent_downloads=50)

    def load_wsi_name_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data
    
    def crop_image(self, row_image, x, slice_size):
        patch_image = row_image.crop((x, 0, x + slice_size[0], slice_size[1]))
        return patch_image

    def parallel_cropper(self, row_image, width, slice_size):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            params = [(row_image, x, slice_size) for x in range(0, width, slice_size[0])]
            patch_img_list = list(executor.map(lambda p: self.crop_image(*p), params))
        return patch_img_list

    def generate_wsi_row_url_info(self, args):
        wsi_name, x, y, width, height, level = args
        return {
            "name": wsi_name,
            "x": str(x),
            "y": str(y),
            "width": str(width),
            "height": str(height),
            "level": str(level)
        }

    def generate_wsi_row_url_infos_concurrently(self, wsi_name, slide_info, num_level, self_max_width, slice_size):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_wsi_info = []
            
            for level in range(num_level):
                width = int(slide_info[f"openslide.level[{level}].width"])
                height = int(slide_info[f"openslide.level[{level}].height"])

                redundant = (width - 1) % self_max_width + 1
                for y in range(0, height, slice_size[1]):
                    for x in range(0, width, self.max_width):
                        width_val = min(self.max_width, redundant)
                        future_to_wsi_info.append(
                            (wsi_name, x, y, width_val, slice_size[1], level)
                        )
            
            results = list(executor.map(self.generate_wsi_row_url_info, future_to_wsi_info))
            
        return results
        
    async def encoding_wsi(self, wsi_name):
        wsi_url = f" metaservice/api/region/openslide/{wsi_name}"
        # wsi_url = f"wsi/metaservice/api/region/openslide/{wsi_name}"
        wsi_info_url = wsi_url.replace("region", "sliceInfo")
        try:
            slide_info = eval(requests.get(wsi_info_url, stream=True).content)
        except:
            print(f"Can not find the information of {wsi_info_url}.")
            return [], []
        slice_size = (256, 256)
        num_level = int(slide_info["openslide.level-count"])

        wsi_row_url_infos = self.generate_wsi_row_url_infos_concurrently(wsi_name, slide_info, num_level, self.max_width, slice_size)
    
        row_image_and_infos = await self.image_downloader.download_images(wsi_row_url_infos)

        wsi_row_infos = []      #   row image   infos 
        wsi_embeddings = []     #   row image   embeddings
        for row_image_and_info in tqdm(row_image_and_infos):

            if row_image_and_info is None:
                continue         

            row_image, row_infos = row_image_image_and_info                                                       
            patch_width = int(row_infos['width'])

            patch_img_list = self.parallel_cropper(row_image, patch_width, slice_size)          #   row image to patch
            embeddings = self.image_embed_model._get_image_list_embeddings(patch_img_list)           #   UNI  
            wsi_embeddings.append(embeddings)
            wsi_row_infos.append(row_infos)

        return wsi_row_infos, wsi_embeddings


app = Flask(__name__)

image_url_downloader = None

def get_image_url_downloader():
    global image_url_downloader
    if image_url_downloader is None:
        image_url_downloader = WSI_Image_Encoder_server()
    return image_url_downloader

@app.route('/wsi_image_embeddings', methods=['POST'])
def process_text2text_retrieval():
    wsi_name = request.json.get('wsi_name')

    image_url_downloader = get_image_url_downloader()
    wsi_row_infos, wsi_embeddings = asyncio.run(image_url_downloader.encoding_wsi(wsi_name))

    return jsonify({
        'wsi_infos':wsi_row_infos,
        'embeddings': wsi_embeddings
    })



if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port="9999")