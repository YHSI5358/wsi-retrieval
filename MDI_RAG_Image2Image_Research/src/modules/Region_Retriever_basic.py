import os, sys, torch, timm, requests
sys.path.append(".")
sys.path.append("..")
sys.path.append("...")
from PIL import Image
from io import BytesIO
from collections import deque, defaultdict
from MDI_RAG_Image2Image_Research.src.utils.retriever import Image2Image_Retriever_Qdrant



class Basic_Region_Retriever():
    def __init__(self, ):
        self.retriever = Image2Image_Retriever_Qdrant()

    def mesh_slides(self, image):
        width, height = image.size
        width_step = 224 - (224 - width % 224) // (width // 224)
        height_step = 224 - (224 - height % 224) // (height // 224)

        image_patches = []
        for x in range(0, width-223, width_step):
            for y in range(0, height-223, height_step):
                cropped_image = image.crop((x, y, x+224, y+224))
                image_patches.append(cropped_image)

        return image_patches
    
    def single_retrieval(self, query_image):
        results = self.retriever.retrieve(query_image, top_k=20)
        return [(result.score, result.payload) for result in results]  
    
    def mesh_slides(self, image):
        width, height = image.size
        width_step = 224 - (224 - width % 224) // (width // 224)
        height_step = 224 - (224 - height % 224) // (height // 224)

        image_patches = []
        for x in range(0, max(1, width-223), width_step):
            for y in range(0, max(1, height-223), height_step):
                cropped_image = image.crop((x, y, x+224, y+224))
                image_patches.append(cropped_image)

        return image_patches
    
    def find_most_wsi_name(self, raw_results):
        score_hist = defaultdict(float)
        result_hist = defaultdict(list)
       
        for result in raw_results:
           for score, payload in result:
               wsi_name = payload["wsi_name"]
               score_hist[wsi_name] += score
               result_hist[wsi_name].append((score, payload))
        
        target = max(score_hist, key=score_hist.get)
        return target, result_hist[target]
    
    def find_region(self, target_results):
        def is_adjacent(rect1, rect2):
            x1, y1, w1, h1 = rect1
            x2, y2, w2, h2 = rect2

            if x1 + w1 == x2 and y1 < y2 + h2 and y2 < y1 + h1:
                return True
            elif x2 + w2 == x1 and y1 < y2 + h2 and y2 < y1 + h1:
                return True
            elif y1 + h1 == y2 and x1 < x2 + w2 and x2 < x1 + w1:
                return True
            elif y2 + h2 == y1 and x1 < x2 + w2 and x2 < x1 + w1:
                return True
            else:
                return False
        
        rect_list = [
            (result[0], [
                int(result[1]['position'][0]) * (2 ** int(result[1]['level'])), 
                int(result[1]['position'][1]) * (2 ** int(result[1]['level'])), 
                int(result[1]['patch_size'][0]) * (2 ** int(result[1]['level'])),  
                int(result[1]['patch_size'][1]) * (2 ** int(result[1]['level'])), 
              ])
            for result in target_results
        ]   
            
        score_results = defaultdict(list)
        region_results = defaultdict(list)

        checked_index = []
        target_deque = deque()   
        for i in range(len(rect_list)):
            if i in checked_index:
                continue
            target_deque.append([i, rect_list[i]])
            checked_index.append(i)

            while len(target_deque) != 0:
                index, cur = target_deque.popleft()
                score1, rect1 = cur[0], cur[1]
                
                score_results[i].append(score1)
                region_results[i].append(rect1)

                for j in range(len(rect_list)):
                    if j in checked_index:
                        continue
                
                    _, rect2 = rect_list[j]
                    if is_adjacent(rect1, rect2):
                        target_deque.append([j, rect_list[j]])
                        checked_index.append(j)
        
        print(region_results)

        regions = []
        for key in region_results:
            cur_patches = region_results[key]
    
            result_x = min([res[0] for res in cur_patches])
            result_y = min([res[1] for res in cur_patches])
            result_w = max([res[0]+res[2] for res in cur_patches]) - result_x
            result_h = max([res[1]+res[3] for res in cur_patches]) - result_y

            target_region = [result_x, result_y, result_w, result_h]
            regions.append(target_region)

        return regions

    def redifine_region(self, target_region, ratio):
        x, y, width, height = target_region
        mid_x = x + width // 2
        mid_y = y + height // 2

        redifine_width = int((width * height * ratio) ** 0.5)  
        redifine_height = int(redifine_width / ratio)
        redifine_x = max(0, mid_x - redifine_width // 2)
        redifine_y = max(0, mid_y - redifine_height // 2)

        return [redifine_x, redifine_y, redifine_width, redifine_height]

    def retrieve(self, image):
        width, height = image.size
        image_patches = self.mesh_slides(image)
        raw_results = [self.single_retrieval(patch) for patch in image_patches]

        target_wsi_name, target_results = self.find_most_wsi_name(raw_results)   
        region_results = self.find_region(target_results)
        
        redifine_region = [self.redifine_region(region, width/height) for region in region_results]

        return target_wsi_name, redifine_region

    

if __name__ == "__main__":
    query_img_path = ""
    if "http" in query_img_path:
        response = requests.get(query_img_path)
        query_image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        query_image = Image.open(query_img_path).convert("RGB")

    retriever = Basic_Region_Retriever()
    wsi_name, results = retriever.retrieve(query_image)

    from MDI_RAG_Image2Image_Research.src.test.cosine_sim import Cosine_Sim_Evaluater
    eva = Cosine_Sim_Evaluater()

    for result in results:
        result_url = f""
        cos_sim = eva.calculate_similarity(query_img_path, result_url)
        print(result_url, cos_sim)