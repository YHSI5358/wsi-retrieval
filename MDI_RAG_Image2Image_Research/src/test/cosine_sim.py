import os, sys, torch, timm, requests
sys.path.append(".")
sys.path.append("..")
sys.path.append("...")
from io import BytesIO
from PIL import Image
import numpy as np
from MDI_RAG_Image2Image_Research.src.utils.encoder import WSI_Image_UNI_Encoder


class Cosine_Sim_Evaluater():
    def __init__(self):
        self.wsi_patch_encoder = WSI_Image_UNI_Encoder()

    def load_img_url(self, img_url):
        if "http" in img_url:
            response = requests.get(img_url)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(img_url).convert("RGB")

        return image


    def calculate_similarity(self, query_url, retrieval_url):
        query_image = self.load_img_url(query_url)
        retrieval_image = self.load_img_url(retrieval_url)

        query_embedding = self.wsi_patch_encoder.encode_image(query_image)
        retrieval_embedding = self.wsi_patch_encoder.encode_image(retrieval_image)

        dot_product = np.dot(query_embedding, retrieval_embedding)

        norm_a = np.linalg.norm(query_embedding)
        norm_b = np.linalg.norm(retrieval_embedding)

        cosine_similarity = dot_product / (norm_a * norm_b)

        return cosine_similarity
    

if __name__ == "__main__":
    eva = Cosine_Sim_Evaluater()
    query_url = ""
    retrieval_url = ""
    cos_sim = eva.calculate_similarity(query_url, retrieval_url)
    print(cos_sim)