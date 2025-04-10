import os, sys, json, asyncio, aiohttp, aiofiles, requests, uuid, logging, timm
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
from torch.utils.data import DataLoader, Dataset
import torch.multiprocessing as mp



class CustomWSIDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.transform = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                         ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


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
        self.embed_model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
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
        # wsi_dataset = CustomWSIDataset(image_paths)
        # dataloader = DataLoader(wsi_dataset, batch_size=16, shuffle=False, num_workers=64, pin_memory=True)

        embeddings = []
        with torch.no_grad():
            for images in tqdm(dataloader, desc=f"WSI name: {wsi_name}", ascii=True):
                images = images.to(self._device)
                embedding = self.embed_model(images)
                embeddings.append(embedding.cpu())

        return embeddings



class ImagePatchEncoder():
    def __init__(self,):
        self.image_encoder = WSI_Image_UNI_Encoder()
        self.loaded_wsi_name_path = "Retrieval_Server/cache/wsi_patch_image/loaded_wsis.json"
        self.image_names = self.load_wsi_name(self.loaded_wsi_name_path)

    def load_wsi_name(self, json_file_path):
        """读取已经缓存过的 WSI Names。"""
        with open(json_file_path, "r") as file:
            try:
                return json.load(file)
            except json.JSONDecodeError:
                return []

    def load_image(self, image_path):
        try:
            img = Image.open(image_path).convert('RGB')
            return img
        except Exception as e:
            print(f"Failed to load image {image_path}: {e}")
            return None

    def get_wsi_patch_info_and_emb(self, wsi_name, dataloader):
        # if wsi_name not in self.image_names:
        #     print(f"WSI {wsi_name} is not loaded.")
        #     return [], []
        
        # folder_path = os.path.join("Retrieval_Server/cache/wsi_patch_image", wsi_name)
        # patch_infos = [filename for filename in os.listdir(folder_path)]     
        # image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        all_embeddings = self.image_encoder.encode_wsi_patch(wsi_name, dataloader)       # list of embeddings (batch_size, 1024)
        patch_embeddings = torch.cat(all_embeddings, dim=0).cpu().tolist()

        return patch_embeddings




if __name__ == "__main__":
    encoder = ImagePatchEncoder()
    patch_embeddings = encoder.get_wsi_patch_info_and_emb("241183-21.tiff")
    print(len(patch_embeddings))