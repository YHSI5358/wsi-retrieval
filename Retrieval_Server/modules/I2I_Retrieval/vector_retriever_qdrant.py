import os, sys, qdrant_client, timm, requests
sys.path.append(".")
sys.path.append("..")
sys.path.append("...")
from PIL import Image
from io import BytesIO
from qdrant_client.http import models as rest
import torch
from torchvision import transforms


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
        self.embed_model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu", weights_only=False), strict=True)
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

    def encode_image(self, image):
        image = self.transform(image).unsqueeze(dim=0).to(self._device)
        with torch.no_grad():
            feature_emb_tensor = self.embed_model(image)
            return feature_emb_tensor.squeeze().tolist()



class Image2Image_Retriever_Qdrant():
    def __init__(self):
        self.image_client_name = "WSI_Region_Retrieval"
        # embed_cache_path = os.path.join("Retrieval_Server/cache", "vector_database", "image2image_retrieval_big_500w_all")
        embed_cache_path = "/hpc2hdd/home/ysi538/retrieval/MDI_RAG_Image2Image_Research/data/vector_database"
        self.image_client = self.init_image_vector_db_client(embed_cache_path)

        # get info
        nums = self.image_client.count(
            collection_name = self.image_client_name,
            exact=True
        )
        print("Number of vectors:", nums)

        self.image_encoder = WSI_Image_UNI_Encoder()

    def init_image_vector_db_client(self, embed_cache_path):
        image_client = qdrant_client.QdrantClient(path=embed_cache_path)
        if not image_client.collection_exists(self.image_client_name):
            image_client.create_collection(
                collection_name=self.image_client_name,
                vectors_config=rest.VectorParams(size=1024, distance=rest.Distance.COSINE),
                on_disk_payload=True,
            )

        return image_client

    def retrieve(self, query_img_url, top_k=20):

        if "http" in query_img_url:
            response = requests.get(query_img_url, verify=False)
            query_image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            query_image = Image.open(query_img_url).convert("RGB")

        query_embedding = self.image_encoder.encode_image(query_image)


        retrieval_results = self.image_client.search(
            collection_name=self.image_client_name,
            query_vector=query_embedding,
            limit=top_k,
            score_threshold=0.4,
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