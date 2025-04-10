import os, sys, json, qdrant_client, requests
sys.path.append(".")
sys.path.append("..")
sys.path.append("...")
from io import BytesIO
from PIL import Image
from llama_index.core import Settings, StorageContext
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from Retrieval_Server.modules.init_database import Initial_Database
from Retrieval_Server.modules.embeddings import Custom_UNI_Image_Embedding, Custom_MPNet_Text_Embedding


class Image2Image_Retriever():
    def __init__(self, database):
        self.embed_cache_path = os.path.join("Retrieval_Server/caches", "vector_database", "image2image_retrieval")

        self.text_embed_model = Custom_MPNet_Text_Embedding(max_length=512)
        self.image_embed_model = Custom_UNI_Image_Embedding()
        Settings.embed_model = self.text_embed_model

        image_nodes = database.image_nodes
        self.image_vector_index = self.load_image_vector_stores(image_nodes)
        print("Finish Initialized Vector Databases.")

    def load_image_vector_stores(self, image_nodes):
        """Create a local Qdrant vector store (Vector Based Database)."""
        client = qdrant_client.QdrantClient(path=self.embed_cache_path)
        text_store = QdrantVectorStore(collection_name="text_collection", client=client)
        image_store = QdrantVectorStore(collection_name="image_collection", client=client)
        storage_context = StorageContext.from_defaults(vector_store=text_store, 
                                                       image_store=image_store)
        loaded_json_path = os.path.join(self.embed_cache_path, "loaded_nodes.json")

        if image_store._collection_exists("image_collection") and os.path.exists(loaded_json_path):
            vector_index = MultiModalVectorStoreIndex.from_vector_store(
                vector_store=text_store,
                embed_model=self.text_embed_model,
                image_vector_store=image_store,
                image_embed_model=self.image_embed_model
            )
            with open(loaded_json_path, "r") as json_file:
                self.loaded_documents = json.load(json_file)
        else:
            vector_index = MultiModalVectorStoreIndex(image_nodes, 
                                                      embed_model=self.text_embed_model, 
                                                      image_embed_model=self.image_embed_model, 
                                                      storage_context=storage_context,
                                                      show_progress=True)
            vector_index.storage_context.persist(self.embed_cache_path)
            self.loaded_nodes = {node.id_:1 for node in image_nodes}
            with open(loaded_json_path, "w") as json_file:
                json.dump(self.loaded_nodes, json_file)
        return vector_index

    def retrieve(self, query_img_url, top_k=20):
        retriever_engine = self.image_vector_index.as_retriever(image_similarity_top_k=top_k)

        query_img_path = "Retrieval_Server/caches/query_image.png"
        if "http" in query_img_url:
            response = requests.get(query_img_url)
            query_img = Image.open(BytesIO(response.content))
            query_img.save(query_img_path)
        else:
            query_img_path = query_img_url

        retrieval_results = retriever_engine.image_to_image_retrieve(query_img_path)
        retrieved_images_nodes = [res.node for res in retrieval_results]
        return retrieved_images_nodes
    



if __name__ == "__main__":
    struc_file = "Retrieval_Server/data/struc.sql"
    data_file = "Retrieval_Server/data/data.sql"
    database = Initial_Database(struc_file, data_file)
    retriever = Image2Image_Retriever(database)

    # query_img_path = "data/Image2Image_Basic_Retrieval/med_4class/fibrous-tissue_2.png"
    query_img_path = "  /images/data-manager/1719375153-105291.png"
    retrieved_images_nodes = retriever.retrieve(query_img_path, top_k=3)
    for node in retrieved_images_nodes:
        print("Image_url", node.image_path)
        print("Position: ", node.metadata["position"])
        print("Level: ", node.metadata["level"])
        print("Size: ", node.metadata["size"])
        print("wsi_image_source: ", node.metadata["wsi_image_source"], "\n")