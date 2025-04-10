import sys, requests, mahotas, random, json
sys.path.append(".")
sys.path.append("..")
sys.path.append("...")
import numpy as np
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from skimage import color
from MDI_RAG_Image2Image_Research.src.utils.encoder import WSI_Image_UNI_Encoder



class Rotation_Invarient_Evaluater():
    def __init__(self):
        self.wsi_patch_encoder = WSI_Image_UNI_Encoder()
        self.url_head = ""

    def load_img_url(self, img_url, angle=0):
        if "http" in img_url:
            data = json.dump({
                "angle":angle
            })
            response = requests.get(img_url, data=data)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(img_url).convert("RGB")

        return image
    
    def load_rotate_img(self, x, y, level, theta):
        new_x = x - 112
        new_y = y - 112

        new_img_url = self.url_head + f"241183-21.tiff/{new_x}/{new_y}/448/448/{level}"
        image = self.load_img_url(new_img_url)

        rotated_img = image.rotate(theta, expand=True)
        width, height = rotated_img.size
        center_x, center_y = width // 2, height // 2

        left = center_x - 112
        top = center_y - 112
        right = center_x + 112
        bottom = center_y + 112
        cropped_img = rotated_img.crop((left, top, right, bottom))

        return cropped_img

    def compute_zernike_moments(self, image):
        gray_image = color.rgb2gray(image)  # 
        radius = min(image.size) // 2  #
        zernike_moments = mahotas.features.zernike_moments(gray_image, radius)

        return zernike_moments
    
    def computer_semantic_feature(self, image):
        image_embedding = self.wsi_patch_encoder.encode_image(image)

        return image_embedding
    
    def calculate_similarity(self, query_embedding, retrieval_embedding):
        dot_product = np.dot(query_embedding, retrieval_embedding)
        norm_a = np.linalg.norm(query_embedding)
        norm_b = np.linalg.norm(retrieval_embedding)
        cosine_similarity = dot_product / (norm_a * norm_b)

        return cosine_similarity

    def rotation_embedding_compare(self, x, y, level, theta):
        image_url = self.url_head + f"241183-21.tiff/{x}/{y}/224/224/{level}"
        image = self.load_img_url(image_url)
        image.save("test1.png")
        image_embedding = self.computer_semantic_feature(image)

        rotate_image = self.load_rotate_img(x, y, level, theta)
        rotate_image.save("test1_rotate.png")
        rotate_embedding = self.computer_semantic_feature(rotate_image)

        sim_score = self.calculate_similarity(image_embedding, rotate_embedding)
        return sim_score


    def rotation_zernike_compare(self, x, y, level, theta):
        image_url = self.url_head + f"241183-21.tiff/{x}/{y}/224/224/{level}"
        image = self.load_img_url(image_url)
        image.save("test1.png")
        image_embedding = self.compute_zernike_moments(image)

        rotate_image = self.load_rotate_img(x, y, level, theta)
        rotate_image.save("test1_rotate.png")
        rotate_embedding = self.compute_zernike_moments(rotate_image)

        sim_score = self.calculate_similarity(image_embedding, rotate_embedding)
        return sim_score

    def random_compare(self, wsi_name):
        wsi_url = self.url_head + wsi_name
        wsi_info_url = wsi_url.replace("region", "sliceInfo")

        slide_info = eval(requests.get(wsi_info_url).content)
        num_level = int(slide_info["openslide.level-count"])

        for level in range(num_level):
            semantic_sim = 1
            zernike_sim = 1
            width = int(slide_info[f"openslide.level[{level}].width"])
            height = int(slide_info[f"openslide.level[{level}].height"])
            for _ in tqdm(range(100)):
                x = random.randint(0, width-224)
                y = random.randint(0, height-224)
                theta = random.randint(0, 360)

                try:
                    semantic_sim = min(semantic_sim, self.rotation_embedding_compare(x, y, level, theta))
                    zernike_sim = min(zernike_sim, self.rotation_zernike_compare(x, y, level, theta))
                except:
                    continue

            print(f"Minimal Embedding Sim Score at level {level}:", semantic_sim)
            print(f"Minimal Zernike Sim Score at level {level}:", zernike_sim)



if __name__ == "__main__":
    eva = Rotation_Invarient_Evaluater()
    # image_url = ""

    # x, y, level, theta = 2800, 4000, 2, 60

    # embedding_sim_score = eva.rotation_embedding_compare(x, y, level, theta)
    # print("Embedding Sim Score: ", embedding_sim_score)

    # zernike_sim_score = eva.rotation_zernike_compare(x, y, level, theta)
    # print("Zernike Sim Score: ",zernike_sim_score)

    # eva.random_compare("241183-21.tiff")


    url1 = ""
    image1 = eva.load_img_url(url1)
    image2 = eva.load_img_url(url1, angle=50)
    image1.save("image1,png")
    image2.save("image2,png")