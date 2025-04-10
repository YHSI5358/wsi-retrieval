import asyncio
import concurrent.futures
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from datetime import datetime
import openslide
import os, sys, json, aiohttp, requests, logging
from openslide.deepzoom import DeepZoomGenerator
import hashlib
import numpy as np
import timm, torch

from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import multiprocessing as mp


class CustomWSIDataset(Dataset):
    def __init__(self, images, transform):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image
    

class WSIUNIEncoder():
    def __init__(self, **kwargs):
        self.embed_model =  timm.create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )
        self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])

        local_dir = "/hpc2hdd/home/ysi538/retrieval/checkpoints/vit_large_patch16_224.dinov2.uni_mass100k/"
        self._device = self.infer_torch_device()
        self.embed_model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu", weights_only=True), strict=True)
        self.embed_model = self.embed_model.to(self._device)
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

    def encode_wsi_patch(self, wsi_name, dataloader):
        embeddings = []
        with torch.no_grad():
            for images in tqdm(dataloader, desc=f"WSI name: {wsi_name}", ascii=True):
                images = images.to(self._device)
                embedding = self.embed_model(images)
                embeddings.append(embedding.cpu())

        if embeddings == []:
            return []
        else:
            patch_embeddings = torch.cat(embeddings, dim=0).cpu().tolist()
            return patch_embeddings
        
    def encode_image(self, patch_image):
        patch_image = self.transform(patch_image).unsqueeze(dim=0).to(self._device)
        embedding = self.embed_model(patch_image)

        return embedding.cpu().squeeze().tolist()
    

class Embedding_loader():
    def __init__(self):
        self.wsi_patch_encoder = WSIUNIEncoder()
        self.cache_path = "/hpc2hdd/home/ysi538/retrieval/MDI_RAG_Image2Image_Research/data/embedding_cache"
        self.loaded_embeddings = os.listdir(self.cache_path)


    def loading_wsi_image(self, wsi_name, images):
        """在 CPU 上获取 WSI patch 的 Dataloader。"""
        wsi_dataset = CustomWSIDataset(images, self.wsi_patch_encoder.transform)
        dataloader = DataLoader(wsi_dataset, batch_size=16, shuffle=False, num_workers=16, pin_memory=True)
        return dataloader


    def loading(self, wsi_name, images):
        if wsi_name in self.loaded_embeddings:
            print(f"WSI {wsi_name} cached.")
            return wsi_name, []
        else:
            dataloader = self.loading_wsi_image(wsi_name, images)
            return wsi_name, dataloader
        
    def encoding(self, wsi_name, patch_infos, dataloader):
        patch_embeddings = self.wsi_patch_encoder.encode_wsi_patch(wsi_name, dataloader)

        dir_path = os.path.join(self.cache_path, wsi_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        info_path = os.path.join(self.cache_path, wsi_name, "patch_info.json")
        with open(info_path, 'w') as file:
            json.dump(patch_infos, file)

        embedding_path = os.path.join(self.cache_path, wsi_name, "embeddings.json")
        with open(embedding_path, 'w') as file:
            json.dump(patch_embeddings, file)


def get_slide_info(wsi_name):
    filepath = os.path.join('caches/wsi_image/', wsi_name)
    filepath = os.path.join('/hpc2hdd/home/ysi538/retrieval/caches/wsi_image/', wsi_name)
    if not os.path.isfile(filepath):
        msg = {"error": "No such file"}
        print(msg)
        return msg
    if os.path.getsize(filepath) < 5 * 1024 * 1024:
        msg = {"error": "File size less than 5MB"}
        return None, None
    try:
        slide = openslide.OpenSlide(filepath)
        slide_properties = slide.properties
    except BaseException as error:
        msg = {"type": "Openslide", "error": str(error)}
        return None, None
        
    return slide, slide_properties


def loading_wsi(wsi_name):
    slice_size = (256, 256)
    slide, slide_info = get_slide_info(wsi_name)
    if slide is None:
        return [], []
    
    num_level = int(slide_info.get('openslide.level-count', 1))
    patch_info_list = []
    for level in range(1, num_level):  # start from level 1
        ratio = slide.level_downsamples[level]
        width = int(slide_info.get(f"openslide.level[{level}].width"))
        height = int(slide_info.get(f"openslide.level[{level}].height"))
        
        for y in range(0, height, slice_size[1]):
            for x in range(0, width, slice_size[0]):
                patch_infos = {
                    "x": int(x * ratio),
                    "y": int(y * ratio),
                    "width": str(slice_size[1]),
                    "height": str(slice_size[0]),
                    "level": str(level)
                }
                patch_info_list.append(patch_infos)

    captions = []
    regions = []
    for patch_info in patch_info_list:
        try:
            x, y, width, height, level = int(patch_info["x"]), int(patch_info["y"]), int(patch_info["width"]), int(patch_info["height"]), int(patch_info["level"])
            region = slide.read_region((x, y), level, (width, height))
            region = region.convert("RGB")
            regions.append(region)
            captions.append(f"_{x}_{y}_{width}_{height}_{level}.png")
        except BaseException as error:
            print(error)
            continue
        
    return regions, captions


async def process_wsi(emb_loader, wsi_name):
    regions, captions = await asyncio.get_event_loop().run_in_executor(None, loading_wsi, wsi_name)
    if regions == [] or captions == []:
        return
    wsi_name, dataloader = emb_loader.loading(wsi_name, regions)
    emb_loader.encoding(wsi_name, captions, dataloader)


async def main():
    wsi_name_list = []
    wsi_dir = "/hpc2hdd/home/ysi538/retrieval/caches/wsi_image/"
    wsi_name_list = os.listdir(wsi_dir)
    emb_loader = Embedding_loader()

    exist_list = emb_loader.loaded_embeddings

    tasks = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for wsi_name in tqdm(wsi_name_list):
            if wsi_name in exist_list:
                continue
            tasks.append(asyncio.create_task(process_wsi(emb_loader, wsi_name)))

        await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())