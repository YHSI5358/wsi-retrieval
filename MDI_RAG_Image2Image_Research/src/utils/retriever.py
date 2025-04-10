import sys, qdrant_client, requests
sys.path.append(".")
sys.path.append("..")
sys.path.append("...")
from PIL import Image
from io import BytesIO
from MDI_RAG_Image2Image_Research.src.utils.encoder import WSI_Image_UNI_Encoder
import time

class Image2Image_Retriever_Qdrant():
    def __init__(self):
        self.image_client_name = "WSI_Region_Retrieval_500k"
        database_path = "MDI_RAG_Image2Image_Research/data/vector_database"  
        self.image_client = qdrant_client.QdrantClient(path=database_path)
        nums = self.image_client.count(collection_name=self.image_client_name)
        print("Number of vectors:", nums)

        self.image_encoder = WSI_Image_UNI_Encoder()

    def retrieve(self, query_embedding, top_k=20):

        begin_time = time.time()
        retrieval_results = self.image_client.search(
            collection_name=self.image_client_name,
            query_vector=query_embedding,
            limit=top_k,
            score_threshold=0,
        )
        end_time = time.time()
        print("Retrieval time:", end_time - begin_time)

        return retrieval_results, end_time - begin_time
    
    def search_multi_imgs(self, query_pathes, m,n, top_k=20):
        cost_list = []

        for query_path in query_pathes:
            query_image = self.request_image(query_path)
            query_embedding = self.image_encoder.encode_image(query_image)
            retrieval_results, cost = self.retrieve(query_embedding, top_k)
            cost_list.append(cost)

        combine_time = time.time()
        connected_regions = self.get_combined_regions(retrieval_results)
        combine_time_cost = time.time() - combine_time
        cost_list = [cost + combine_time_cost for cost in cost_list]
        return cost_list, connected_regions
    
    def get_combined_regions(self, retrieval_results):

        # 18632_L02_CD4.svs_0_0_256_256_1.png格式18632_L02_CD4.svs_14336_1536_256_256_1.png
        search_info_list = []
        
        for result in retrieval_results:
            payload = result.payload
            search_info = {}
            level = payload['level']
            w = payload['position'][0]
            h = payload['position'][1]
            x = payload['patch_size'][0]
            y = payload['patch_size'][1]
            id = payload['wsi_name'].split("_")[0]
            name = "_".join(payload['wsi_name'].split("_")[1:])
            search_info = {"id": id, "name": name, "x": x, "y": y, "w": w, "h": h, "level": level}
            search_info_list.append(search_info)

        def dfs(node, component, visited):

            visited[node] = True

            component.append(search_info_list[node])
            
            for neighbor in range(len(search_info_list)):
                if not visited[neighbor] and self.judge_if_connected(search_info_list[node], search_info_list[neighbor]):

                    dfs(neighbor, component, visited)


        visited = [False] * len(search_info_list)
        components = []


        for i in range(len(search_info_list)):
            if not visited[i]:

                current_component = []

                dfs(i, current_component, visited)

                components.append(current_component)


        components = [component for component in components if len(component) > 1]

        return components
    
    def judge_if_connected(self, info1, info2):

        if info1["name"] != info2["name"]:
            return False
        if info1["level"] != info2["level"]:
            return False
        if int(info1["x"]) + int(info1["w"]) == int(info2["x"]) and int(info1["y"]) == int(info2["y"]):
            return True
        if int(info1["x"]) - int(info1["w"]) == int(info2["x"]) and int(info1["y"]) == int(info2["y"]):
            return True
        if int(info1["y"]) + int(info1["h"]) == int(info2["y"]) and int(info1["x"]) == int(info2["x"]):
            return True
        if int(info1["y"]) - int(info1["h"]) == int(info2["y"]) and int(info1["x"]) == int(info2["x"]):
            return True
        if int(info1["x"]) + int(info1["w"]) == int(info2["x"]) and int(info1["y"]) + int(info1["h"]) == int(info2["y"]):
            return True
        if int(info1["x"]) - int(info1["w"]) == int(info2["x"]) and int(info1["y"]) - int(info1["h"]) == int(info2["y"]):
            return True
        if int(info1["x"]) + int(info1["w"]) == int(info2["x"]) and int(info1["y"]) - int(info1["h"]) == int(info2["y"]):
            return True
        if int(info1["x"]) - int(info1["w"]) == int(info2["x"]) and int(info1["y"]) + int(info1["h"]) == int(info2["y"]):
            return True
        return False
    
    def request_image(self, query_img_path):
        if "http" in query_img_path:
            response = requests.get(query_img_path, verify=False)
            query_image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            query_image = Image.open(query_img_path).convert("RGB")
        return query_image


if __name__ == "__main__":
    retriever = Image2Image_Retriever_Qdrant()

    query_img_path =  ""


    cost_list = []
    m=1
    n=1

    cost_list, connected_regions = retriever.search_multi_imgs([query_img_path],m,n, top_k=20)

    print(f"Average search cost: {sum(cost_list) / len(cost_list)}")
    print(f"Connected regions: {len(connected_regions)}")

