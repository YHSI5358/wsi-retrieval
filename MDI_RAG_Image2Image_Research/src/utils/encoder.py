import os
import sys
import torch
import timm
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset
import logging


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
        try:
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            logger.error(f" {self.image_paths[idx]} : {str(e)}")
            
            blank_image = torch.zeros(3, 224, 224)
            return blank_image

class WSI_Image_UNI_Encoder:

    def __init__(self, model_dir=None):

        
        try:
            self.embed_model = timm.create_model(
                "vit_large_patch16_224", 
                img_size=224, 
                patch_size=16, 
                init_values=1e-5, 
                num_classes=0, 
                dynamic_img_size=True
            )
            
            
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])

            
            if model_dir is None:
                model_dir = "/hpc2hdd/home/retrieval/checkpoints/vit_large_patch16_224.dinov2.uni_mass100k/"
                if not os.path.exists(model_dir):
                    model_dir = "checkpoints/vit_large_patch16_224.dinov2.uni_mass100k/"
            
            
            self._device = self._infer_torch_device()
            logger.info(f"{self._device}")
            
            
            if os.path.exists(os.path.join(model_dir, "pytorch_model.bin")):
                self.embed_model.load_state_dict(
                    torch.load(
                        os.path.join(model_dir, "pytorch_model.bin"), 
                        map_location="cpu", 
                        weights_only=True
                    ), 
                    strict=True
                )
                self.embed_model = self.embed_model.to(self._device)
                self.embed_model.eval()
                logger.info(f" {model_dir} ")
            else:
                logger.error(f": {os.path.join(model_dir, 'pytorch_model.bin')}")
                raise FileNotFoundError(f": {os.path.join(model_dir, 'pytorch_model.bin')}")
        
        except Exception as e:
            logger.error(f": {str(e)}")
            raise

    def _infer_torch_device(self):

        try:
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        except Exception as e:
            logger.warning(f" {str(e)}")
            return "cpu"

    def encode_image(self, patch_image):

        try:
            patch_image = self.transform(patch_image).unsqueeze(dim=0).to(self._device)
            with torch.no_grad():
                embedding = self.embed_model(patch_image)
            return embedding.cpu().squeeze().tolist()
        except Exception as e:
            logger.error(f"{str(e)}")
            
            return [0.0] * 1024

    def encode_wsi_patch(self, wsi_name, dataloader, batch_size=32):

        embeddings = []
        try:
            with torch.no_grad():
                for images in tqdm(dataloader, desc=f"WSI name: {wsi_name}", ascii=True):
                    images = images.to(self._device)
                    embedding = self.embed_model(images)
                    embeddings.append(embedding.cpu())
            return embeddings
        except Exception as e:
            logger.error(f"{str(e)}")
            return embeddings
