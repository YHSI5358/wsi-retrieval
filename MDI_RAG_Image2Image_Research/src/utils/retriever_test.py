import sys, qdrant_client, requests
sys.path.append(".")
sys.path.append("..")
sys.path.append("...")
from PIL import Image
from io import BytesIO
from MDI_RAG_Image2Image_Research.src.utils.encoder import WSI_Image_UNI_Encoder
import time
from qdrant_client.http import models as rest

class Image2Image_Retriever_Qdrant():
    def __init__(self):
        self.image_client_name = "WSI_Region_Retrieval"  
        database_path = "MDI_RAG_Image2Image_Research/data/vector_database"  
        self.image_client = qdrant_client.QdrantClient(path=database_path)
        
        
        self.new_collection_name = "WSI_Region_Retrieval_500k"
        
        
        
        
        

        
        original_vectors, _ = self.image_client.scroll(
            collection_name=self.image_client_name,
            limit=500000
        )
        print("Number of original vectors:", len(original_vectors))
        
        
        self.image_client.upsert(
            collection_name=self.new_collection_name,
            points=original_vectors
        )
        
        nums = self.image_client.count(collection_name=self.new_collection_name)
        print("Number of vectors:", nums)

        self.image_encoder = WSI_Image_UNI_Encoder()

    def retrieve(self, image, top_k=20):
        query_embedding = self.image_encoder.encode_image(image)

        begin_time = time.time()
        retrieval_results = self.image_client.search(
            collection_name=self.new_collection_name,
            query_vector=query_embedding,
            limit=top_k,
            score_threshold=0,
        )
        end_time = time.time()
        print("Retrieval time:", end_time - begin_time)

        return retrieval_results, end_time - begin_time


if __name__ == "__main__":
    retriever = Image2Image_Retriever_Qdrant()

    query_img_path =  ""
    
    if "http" in query_img_path:
        response = requests.get(query_img_path, verify=False)
        query_image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        query_image = Image.open(query_img_path).convert("RGB")
    
    print("Query image Path:", query_img_path)
    
    query_image.save("query_image.jpg")

    cost_list = []

    for i in range(10):
        results,time_cost = retriever.retrieve(query_image, top_k=20)
        cost_list.append(time_cost)

    print(f"Average search cost: {sum(cost_list) / len(cost_list)}")
    
    
    