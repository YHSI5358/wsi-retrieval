import os, sys, qdrant_client, timm, requests, cProfile, pstats
sys.path.append(".")
sys.path.append("..")
sys.path.append("...")
from PIL import Image
from io import BytesIO, StringIO
from qdrant_client.http import models as rest
import torch
from torchvision import transforms


class Image2Image_Retriever_Qdrant():
    def __init__(self):
        self.image_client_name = "LBP_WSI_Image"
        embed_cache_path = os.path.join("Retrieval_Server/cache", "vector_database", "image2image_retrieval_big_500w_final")    
        # test "237208-12.tiff", "241183-21.tiff"
        self.image_client = qdrant_client.QdrantClient(path=embed_cache_path)

        print(self.image_client.get_collection(f"{self.image_client_name}"))

        nums = self.image_client.count(collection_name=self.image_client_name)
        print("Number of vectors:", nums)

    def retrieve(self, query_img_url, top_k=20):
        query_embedding = [0.1 for _ in range(1024)]    # 测试 query embedding

        retrieval_results = self.image_client.search(
            collection_name=self.image_client_name,
            query_vector=query_embedding,
            limit=top_k,
            # score_threshold=0.6,
        )

        return retrieval_results


if __name__ == "__main__":
    retriever = Image2Image_Retriever_Qdrant()
    query_img_path =  " metaservice/api/region/openslide/241183-21.tiff/6400/25344/256/256/1"
    results = retriever.retrieve(query_img_path, top_k=20)
    for result in results:
        print(result.payload)


    def main(retriever, query_img_path):
        return retriever.retrieve(query_img_path, top_k=20)

    profile = cProfile.Profile()
    profile.enable()
    main(retriever, query_img_path)
    profile.disable()

    s = StringIO()
    ps = pstats.Stats(profile, stream=s).sort_stats('tottime')
    ps.print_stats(50)
    print(s.getvalue())
