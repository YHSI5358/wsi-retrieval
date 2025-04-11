import requests, heapq
from PIL import Image
from copy import deepcopy
from io import BytesIO
import numpy as np
from MDI_RAG_Image2Image_Research.src.utils.encoder import WSI_Image_UNI_Encoder


def load_img_url(img_url):
    if "http" in img_url:
        response = requests.get(img_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(img_url).convert("RGB")

    return image


class WSI_Quadtree_node():
    def __init__(self, wsi_name, x, y, w, h, level, img_encoder, parent_index=None):
        self.wsi_name = wsi_name
        if w < 224 and level > 0:
            self.x, self.y, self.w, self.h, self.level = x*2, y*2, w*2, h*2, level-1
        else:
            self.x, self.y, self.w, self.h, self.level = x, y, w, h, level
        self.node_index = f"{self.x}/{self.y}/{self.w}/{self.h}/{self.level}"
        self.img_encoder = img_encoder

        self.parent_index = parent_index
        self.leafs = [None, None, None, None]
        self.sim_score = self.get_sim_score()

    def split(self):
        self.leafs[1] = WSI_Quadtree_node(self.wsi_name, self.x, self.y, self.w//2, self.h//2, self.level, self.img_encoder, self.node_index)
        self.leafs[2] = WSI_Quadtree_node(self.wsi_name, self.x+self.w//2, self.y, self.w//2, self.h//2, self.level, self.img_encoder, self.node_index)
        self.leafs[3] = WSI_Quadtree_node(self.wsi_name, self.x, self.y+self.h//2, self.w//2, self.h//2, self.level, self.img_encoder, self.node_index)
        self.leafs[4] = WSI_Quadtree_node(self.wsi_name, self.x+self.w//2, self.y+self.h//2, self.w//2, self.h//2, self.level, self.img_encoder, self.node_index)

    def merge(self):
        self.leafs = [None, None, None, None]

    def get_sim_score_single(self, leaf):
        target_url = f" metaservice/api/region/openslide/{self.wsi_name}/{self.x}/{self.y}/{self.w}/{self.h}/{self.level}"
        leaf_url =  f" metaservice/api/region/openslide/{leaf.wsi_name}/{leaf.x}/{leaf.y}/{leaf.w}/{leaf.h}/{leaf.level}"
        target_embeding = self.img_encoder.encode_image(load_img_url(target_url))
        leaf_embeding = self.img_encoder.encode_image(load_img_url(leaf_url))

        dot_product = np.dot(target_embeding, leaf_embeding)
        norm_a = np.linalg.norm(target_embeding)
        norm_b = np.linalg.norm(leaf_embeding)
        cosine_similarity = dot_product / (norm_a * norm_b)

        return cosine_similarity

    def get_sim_score(self):
        if self.leafs[0] == None:
            self.split()
        
        scores = 0
        for leaf in self.leafs:
            scores += self.get_sim_score_single(leaf)

        return scores / 4


class WSI_Hilbert_Curve_Encoder():
    def __init__(self): 
        self.wsi_patch_encoder = WSI_Image_UNI_Encoder()
        self.url_head = " metaservice/api/region/openslide/"
        self.max_patch_num = 1024

    def init_nodes(self, name, width, height, level):
        width_step = 224 - (224 - width % 224) // (width // 224)
        height_step = 224 - (224 - height % 224) // (height // 224)

        node_dict = {}
        for x in range(0, max(1, width-223), width_step):
            for y in range(0, max(1, height-223), height_step):
                cur_node = WSI_Quadtree_node(name, x, y, 224, 224, level, self.wsi_patch_encoder)
                node_dict[cur_node.node_index] = cur_node

        return node_dict

    def split_iteration(self, leaf_node_dict, total_node_dict):
        leaf_score = []  # leaf_score  
        for index in leaf_node_dict:
            heapq.heappush(leaf_score, (leaf_node_dict[index].sim_score, index))

        while len(total_node_dict) < self.max_patch_num:
            _, node_index = heapq.heappop(leaf_score)
            node = leaf_node_dict[node_index]
            
            node.split()
            del leaf_node_dict[node_index] 

            for leaf in node.leafs:
                leaf_node_dict[leaf.node_index] = leaf
                heapq.heappush(leaf_score, (leaf.sim_score, leaf.node_index))

        parent_score = [] # leaf_score  
        for index in leaf_node_dict:
            if leaf_node_dict[index].sim_score.parent_index is not None:
                parent_item = (total_node_dict[leaf_node_dict[index].sim_score.parent_index].sim_score, leaf_node_dict[index].sim_score.parent_index)
                if parent_item not in parent_score:
                    heapq.heappush(parent_score, parent_item)

        while True:
            if not parent_score or not leaf_score:
                break  #  

            #  
            leaf_min_score, leaf_min_index = leaf_score[0]
            parent_max_score, parent_max_index = parent_score[0]

            #  
            if leaf_min_score < parent_max_score:
                node = node_dict[leaf_min_index]
                node.split()
                del node_dict[leaf_min_index]  #  

                #  
                for leaf in node.leafs:
                    node_dict[leaf.node_index] = leaf
                    heapq.heappush(leaf_score, (leaf.sim_score, leaf.node_index))
            else:
                #  
                break
        

    def load_wsi(self, wsi_name):
        """  Name of WSI   patch  """
        # if wsi_name in self.image_names:
        #     print(f"Patch of WSI {wsi_name} in the Cache.")
        #     return

        wsi_url = self.url_head + wsi_name
        wsi_info_url = wsi_url.replace("region", "sliceInfo")

        try:
            slide_info = eval(requests.get(wsi_info_url).content)
        except:
            print(f"Can not find the information of {wsi_info_url}.")
            return 

        if slide_info == {'error': 'No such file'}:
            print(f"Can not find usrful slide_info :{wsi_info_url}")
            return

        try:
            num_level = int(slide_info["openslide.level-count"])
        except:
            print(f"None useful num_level in wsi {wsi_name}")
            return

        width = int(slide_info[f"openslide.level[{num_level-1}].width"])
        height = int(slide_info[f"openslide.level[{num_level-1}].height"])

        leaf_node_dict = self.init_nodes(wsi_name, width, height, num_level)
        total_node_dict = deepcopy(leaf_node_dict)
        final_nodes = self.split_iteration(leaf_node_dict, total_node_dict)