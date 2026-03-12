import sys, os, torch, timm, json, requests, ast
sys.path.append(".")
sys.path.append("..")
sys.path.append("...")
from typing import Any, List, Optional
from PIL import Image
from torchvision import transforms
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.base.embeddings.base import Embedding
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.utils import get_cache_dir, infer_torch_device
from llama_index.core.constants import DEFAULT_EMBED_BATCH_SIZE
from llama_index.core.embeddings.multi_modal_base import MultiModalEmbedding
from llama_index.core.schema import ImageType
from llama_index.core.callbacks import CallbackManager
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# from Retrieval_Server.modules.load_images import load_svsortiff_image, load_pngorjpg_image
from external_project.CONCH.conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize


# class Custom_MPNet_Text_Embedding(BaseEmbedding):
#     max_length: int = Field(
#         default=512, description="Maximum length of input.", gt=0
#     )
#     normalize: bool = Field(default=True, description="Normalize embeddings or not.")

#     _model: Any = PrivateAttr()
#     _tokenizer: Any = PrivateAttr()
#     _device: str = PrivateAttr()

#     def __init__(
#         self,
#         tokenizer_name: Optional[str] = "deprecated",
#         pooling: str = "deprecated",
#         max_length: Optional[int] = None,
#         query_instruction: Optional[str] = None,
#         text_instruction: Optional[str] = None,
#         normalize: bool = True,
#         model: Optional[Any] = "deprecated",
#         tokenizer: Optional[Any] = "deprecated",
#         embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
#         cache_folder: Optional[str] = None,
#         trust_remote_code: bool = False,
#         device: Optional[str] = None,
#         callback_manager: Optional[CallbackManager] = None,
#         **model_kwargs,
#     ):
#         self._device = device or infer_torch_device()

#         cache_folder = cache_folder or get_cache_dir()

#         for variable, value in [
#             ("model", model),
#             ("tokenizer", tokenizer),
#             ("pooling", pooling),
#             ("tokenizer_name", tokenizer_name),
#         ]:
#             if value != "deprecated":
#                 raise ValueError(
#                     f"{variable} is deprecated. Please remove it from the arguments."
#                 )
        
#         self._tokenizer = AutoTokenizer.from_pretrained('external_project/mpnet_base_v2')
#         self._model = AutoModel.from_pretrained('external_project/mpnet_base_v2').to(self._device)

#         super().__init__(
#             embed_batch_size=embed_batch_size,
#             callback_manager=callback_manager,
#             max_length=max_length,
#             normalize=normalize,
#             query_instruction=query_instruction,
#             text_instruction=text_instruction,
#         )

#     @classmethod
#     def class_name(cls) -> str:
#         return "HuggingFaceEmbedding"
    
#     def mean_pooling(self, model_output, attention_mask):
#         token_embeddings = model_output[0] #First element of model_output contains all token embeddings
#         input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#         return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


#     def _embed(
#         self,
#         sentences: List[str],
#         prompt_name: Optional[str] = None,
#     ) -> List[List[float]]:
#         """Embed sentences."""
#         encoded_input = self._tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(self._device)
#         with torch.no_grad():
#             model_output = self._model(**encoded_input)
#         sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
#         sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

#         return sentence_embeddings.tolist()[0]

#     def _get_query_embedding(self, query: str) -> List[float]:
#         """Get query embedding."""
#         return self._embed(query, prompt_name="")

#     async def _aget_query_embedding(self, query: str) -> List[float]:
#         """Get query embedding async."""
#         return self._get_query_embedding(query)

#     async def _aget_text_embedding(self, text: str) -> List[float]:
#         """Get text embedding async."""
#         return self._get_text_embedding(text)

#     def _get_text_embedding(self, text: str) -> List[float]:
#         """Get text embedding."""
#         return self._embed(text, prompt_name="")

#     def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
#         """Get text embeddings."""
#         return [self._embed(text, prompt_name="") for text in texts]

class Custom_MPNet_Text_Embedding(BaseEmbedding):
    max_length: int = Field(
        default=512, description="Maximum length of input.", gt=0
    )
    normalize: bool = Field(default=True, description="Normalize embeddings or not.")

    _model: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()
    _device: str = PrivateAttr()

    def __init__(
        self,
        tokenizer_name: Optional[str] = "deprecated",
        pooling: str = "deprecated",
        max_length: Optional[int] = None,
        query_instruction: Optional[str] = None,
        text_instruction: Optional[str] = None,
        normalize: bool = True,
        model: Optional[Any] = "deprecated",
        tokenizer: Optional[Any] = "deprecated",
        embed_batch_size: int = 32,
        cache_folder: Optional[str] = None,
        trust_remote_code: bool = False,
        device: Optional[str] = None,
        callback_manager: Optional[Any] = None,
        **model_kwargs,
    ):
        super().__init__(
            max_length=max_length,
            normalize=normalize,
        )
        self._device = device or self.infer_torch_device()

        cache_folder = cache_folder or self.get_cache_dir()

        for variable, value in [
            ("model", model),
            ("tokenizer", tokenizer),
            ("pooling", pooling),
            ("tokenizer_name", tokenizer_name),
        ]:
            if value != "deprecated":
                raise ValueError(
                    f"{variable} is deprecated. Please remove it from the arguments."
                )
        
        self._tokenizer = AutoTokenizer.from_pretrained('/hpc2hdd/home/ysi538/retrieval/external_project/mpnet_base_v2')
        self._model = AutoModel.from_pretrained('/hpc2hdd/home/ysi538/retrieval/external_project/mpnet_base_v2').to(self._device)

    @classmethod
    def class_name(cls) -> str:
        return "HuggingFaceEmbedding"
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _embed(
        self,
        sentences: List[str],
        prompt_name: Optional[str] = None,
    ) -> List[List[float]]:
        """Embed sentences."""
        encoded_input = self._tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(self._device)
        with torch.no_grad():
            model_output = self._model(**encoded_input)
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings.tolist()[0]

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self._embed(query, prompt_name="")

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Get query embedding async."""
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Get text embedding async."""
        return self._get_text_embedding(text)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self._embed(text, prompt_name="")

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        return [self._embed(text, prompt_name="") for text in texts]

    def infer_torch_device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def get_cache_dir(self):
        return "./cache"
    

class Custom_Llama3_Text_Embedding(BaseEmbedding):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.embed_batch_size = 60

    @classmethod
    def class_name(cls) -> str:
        return "Llama3 Text Embedding Model"

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def embedding_texts(self, texts):
        # print(len(texts), texts)
        url = "  /hpc/llama3/tools/v1/embedding/last_hidden_state"
        # url_internal = http://10.90.156.242:16900/v1/embedding/last_hidden_state
        payload = json.dumps({
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "messages": texts,
        "max_length": "4096" #  8192
        })
        headers = {
        'Content-Type': 'application/json'
        }
        response = requests.request("POST", url, headers=headers, data=payload)
        
        return ast.literal_eval(response.text)["embeddings"]

    def _get_query_embedding(self, query: str) -> List[float]:
        return self.embedding_texts([query])[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        return self.embedding_texts([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self.embedding_texts(texts)

class Custom_Conch_Text_Embedding(BaseEmbedding):
    model: Any = Field(
        default=create_model_from_pretrained("conch_ViT-B-16", 
                                             checkpoint_path="/hpc2hdd/home/ysi538/retrieval/checkpoints/conch/CONCH_pytorch_model_vitb16.bin",
                                             return_transform=False)
    )
    tokenizer: Any = Field(
        default=get_tokenizer()
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "Conch Text Embedding Model"

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        text_tokens = tokenize(texts=[query], tokenizer=self.tokenizer)
        embedding = self.model.encode_text(text_tokens)
        return embedding[0].tolist()

    def _get_text_embedding(self, text: str) -> List[float]:
        text_tokens = tokenize(texts=[text], tokenizer=self.tokenizer)
        embedding = self.model.encode_text(text_tokens)
        return embedding[0].tolist()

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        text_tokens = tokenize(texts=texts, tokenizer=self.tokenizer)
        embeddings = self.model.encode_text(text_tokens)
        return embeddings.tolist()
# class Custom_Conch_Text_Embeddimg(BaseEmbedding):
#     model = Field(
#         default = create_model_from_pretrained("conch_ViT-B-16", 
#                                                checkpoint_path="checkpoints/conch/CONCH_pytorch_model_vitb16.bin",
#                                                return_transform=False)
#     )
#     tokenizer = Field(
#         default=get_tokenizer()
#     )
#     def __init__(self, **kwargs: Any) -> None:
#         super().__init__(**kwargs)

#     @classmethod
#     def class_name(cls) -> str:
#         return "Conch Text Embedding Model"

#     async def _aget_query_embedding(self, query: str) -> List[float]:
#         return self._get_query_embedding(query)

#     async def _aget_text_embedding(self, text: str) -> List[float]:
#         return self._get_text_embedding(text)

#     def _get_query_embedding(self, query: str) -> List[float]:
#         text_tokens = tokenize(texts=[query], tokenizer=self.tokenizer)
#         embedding = self.model.encode_text(text_tokens)
#         return embedding[0].tolist()

#     def _get_text_embedding(self, text: str) -> List[float]:
#         text_tokens = tokenize(texts=[text], tokenizer=self.tokenizer)
#         embedding = self.model.encode_text(text_tokens)
#         return embedding[0].tolist()

#     def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
#         text_tokens = tokenize(texts=texts, tokenizer=self.tokenizer)
#         embeddings = self.model.encode_text(text_tokens)
#         return embeddings.tolist()
    


class Custom_Conch_Image_Embedding(MultiModalEmbedding):
    embed_batch_size: int = Field(default=DEFAULT_EMBED_BATCH_SIZE, gt=0)
    embed_model: Any = Field(
        default=create_model_from_pretrained("conch_ViT-B-16", 
                                             checkpoint_path="/hpc2hdd/home/ysi538/retrieval/checkpoints/conch/CONCH_pytorch_model_vitb16.bin",
                                             return_transform=False)
    )
    tokenizer: Any = Field(
        default=get_tokenizer()
    )
    transform: Any = Field(
        default=transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ))

    _clip: Any = PrivateAttr()
    _model: Any = PrivateAttr()
    _preprocess: Any = PrivateAttr()
    _device: Any = PrivateAttr()

    @classmethod
    def class_name(cls) -> str:
        return "custom_conch_image_embedding_model"

    def __init__(self, *, embed_batch_size=1024, **kwargs):
        super().__init__(embed_batch_size=embed_batch_size, **kwargs)
        self.embed_model.eval()

    async def _aget_query_embedding(self, query: str) -> Embedding:
        return self._get_query_embedding(query)

    def _get_query_embedding(self, query: str) -> Embedding:
        try:
            embedding = self._get_image_embedding(query)
        except:
            embedding = self._get_text_embedding(query)
        return embedding
    
    def _get_text_embedding(self, text: str) -> List[float]:
        text_tokens = tokenize(texts=[text], tokenizer=self.tokenizer)
        embedding = self.embed_model.encode_text(text_tokens)
        return embedding[0].tolist()

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        text_tokens = tokenize(texts=texts, tokenizer=self.tokenizer)
        embeddings = self.embed_model.encode_text(text_tokens)
        return embeddings.tolist()

    async def _aget_image_embedding(self, img_file_path: ImageType) -> Embedding:
        return self._get_image_embedding(img_file_path)

    def _get_image_embedding(self, img_file_path) -> Embedding:
        image = Image.open(img_file_path)
        image = self.transform(image).unsqueeze(dim=0) 
        with torch.inference_mode():
            feature_emb_tensor = self.embed_model.encode_image(image, proj_contrast=False, normalize=False)
            return feature_emb_tensor.squeeze().tolist()

# class Custom_Conch_Image_Embedding(MultiModalEmbedding):
#     embed_batch_size: int = Field(default=DEFAULT_EMBED_BATCH_SIZE, gt=0)
#     embed_model = Field(
#         create_model_from_pretrained("conch_ViT-B-16", 
#                                      checkpoint_path="checkpoints/conch/CONCH_pytorch_model_vitb16.bin",
#                                      return_transform=False)
#     )
#     tokenizer = Field(
#         default=get_tokenizer()
#     )
#     transform = Field(
#         default=transforms.Compose(
#         [
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#         ]
#     ))

#     _clip: Any = PrivateAttr()
#     _model: Any = PrivateAttr()
#     _preprocess: Any = PrivateAttr()
#     _device: Any = PrivateAttr()

#     @classmethod
#     def class_name(cls) -> str:
#         return "custom_conch_image_embedding_model"

#     def __init__(self, *, embed_batch_size=1024, **kwargs):
#         super().__init__(embed_batch_size=embed_batch_size, **kwargs)
#         self.embed_model.eval()

#     async def _aget_query_embedding(self, query: str) -> Embedding:
#         return self._get_query_embedding(query)

#     def _get_query_embedding(self, query: str) -> Embedding:
#         try:
#             embedding = self._get_image_embedding(query)
#         except:
#             embedding = self._get_text_embedding(query)
#         return embedding
    
#     def _get_text_embedding(self, text: str) -> List[float]:
#         text_tokens = tokenize(texts=[text], tokenizer=self.tokenizer)
#         embedding = self.embed_model.encode_text(text_tokens)
#         return embedding[0].tolist()

#     def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
#         text_tokens = tokenize(texts=texts, tokenizer=self.tokenizer)
#         embeddings = self.embed_model.encode_text(text_tokens)
#         return embeddings.tolist()

#     async def _aget_image_embedding(self, img_file_path: ImageType) -> Embedding:
#         return self._get_image_embedding(img_file_path)

#     def _get_image_embedding(self, img_file_path) -> Embedding:
#         image = Image.open(img_file_path)
#         image = self.transform(image).unsqueeze(dim=0) 
#         with torch.inference_mode():
#             feature_emb_tensor = self.embed_model.encode_image(image, proj_contrast=False, normalize=False)
#             return feature_emb_tensor.squeeze().tolist()


class Custom_UNI_Image_Embedding(MultiModalEmbedding):
    embed_batch_size: int = Field(default=DEFAULT_EMBED_BATCH_SIZE, gt=0)
    embed_model: Any = Field(
        default= timm.create_model(
        "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
    ))
    transform: Any = Field(
        default=transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ))

    _clip: Any = PrivateAttr()
    _model: Any = PrivateAttr()
    _preprocess: Any = PrivateAttr()
    _device: Any = PrivateAttr()

    @classmethod
    def class_name(cls) -> str:
        return "custom_uni_embedding_model"

    def __init__(self, *, embed_batch_size=1024, **kwargs):
        super().__init__(embed_batch_size=embed_batch_size, **kwargs)
        local_dir = "/hpc2hdd/home/ysi538/retrieval/checkpoints/vit_large_patch16_224.dinov2.uni_mass100k/"
        self._device = infer_torch_device()
        self.embed_model = self.embed_model.to(self._device)
        self.embed_model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
        self.embed_model.eval()

    async def _aget_query_embedding(self, query: str) -> Embedding:
        return self._get_query_embedding(query)

    def _get_query_embedding(self, query: str) -> Embedding:
        return self._get_image_embedding(query)
    
    def _get_text_embedding(self, text: str) -> Embedding:
        return self._get_text_embeddings([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        results = []
        for text in texts:
            try:
                import clip
            except ImportError:
                raise ImportError("ClipEmbedding requires `pip install git+https://github.com/openai/CLIP.git` and torch.")
            text_embedding = self._model.encode_text(clip.tokenize(text).to(self._device))
            results.append(text_embedding.tolist()[0])

        return results

    async def _aget_image_embedding(self, img_file_path: ImageType) -> Embedding:
        return self._get_image_embedding(img_file_path)

    def _get_image_embedding(self, img_file_path) -> Embedding:
        image = Image.open(img_file_path).convert("RGB")
        image = self.transform(image).unsqueeze(dim=0).to(self._device)
        with torch.inference_mode():
            feature_emb_tensor = self.embed_model(image) 
            return feature_emb_tensor.squeeze().tolist()
# class Custom_UNI_Image_Embedding(MultiModalEmbedding):
#     embed_batch_size: int = Field(default=DEFAULT_EMBED_BATCH_SIZE, gt=0)
#     embed_model = Field(
#         default= timm.create_model(
#         "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
#     ))
#     transform = Field(
#         default=transforms.Compose(
#         [
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#         ]
#     ))

#     _clip: Any = PrivateAttr()
#     _model: Any = PrivateAttr()
#     _preprocess: Any = PrivateAttr()
#     _device: Any = PrivateAttr()

#     @classmethod
#     def class_name(cls) -> str:
#         return "custom_uni_embedding_model"

#     def __init__(self, *, embed_batch_size=1024, **kwargs):
#         super().__init__(embed_batch_size=embed_batch_size, **kwargs)
#         local_dir = "checkpoints/vit_large_patch16_224.dinov2.uni_mass100k/"
#         self._device = infer_torch_device()
#         self.embed_model = self.embed_model.to(self._device)
#         self.embed_model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
#         self.embed_model.eval()

#     async def _aget_query_embedding(self, query: str) -> Embedding:
#         return self._get_query_embedding(query)

#     def _get_query_embedding(self, query: str) -> Embedding:
#         return self._get_image_embedding(query)
    
#     def _get_text_embedding(self, text: str) -> Embedding:
#         return self._get_text_embeddings([text])[0]

#     def _get_text_embeddings(self, texts: List[str]) -> List[Embedding]:
#         results = []
#         for text in texts:
#             try:
#                 import clip
#             except ImportError:
#                 raise ImportError("ClipEmbedding requires `pip install git+https://github.com/openai/CLIP.git` and torch.")
#             text_embedding = self._model.encode_text(clip.tokenize(text).to(self._device))
#             results.append(text_embedding.tolist()[0])

#         return results

#     async def _aget_image_embedding(self, img_file_path: ImageType) -> Embedding:
#         return self._get_image_embedding(img_file_path)

#     def _get_image_embedding(self, img_file_path) -> Embedding:
#         image = Image.open(img_file_path).convert("RGB")
#         image = self.transform(image).unsqueeze(dim=0).to(self._device)
#         with torch.inference_mode():
#             feature_emb_tensor = self.embed_model(image) 
#             return feature_emb_tensor.squeeze().tolist()



if __name__ == "__main__":
    import numpy as np
    # custom_embed_model = Custom_MPNet_Text_Embedding(max_length=512)
    custom_embed_model = Custom_Llama3_Text_Embedding()

    text_embedding = custom_embed_model.get_text_embedding(" ")
    print(type(text_embedding), len(text_embedding))
    query_embedding = custom_embed_model.get_query_embedding(" ")
    print(type(query_embedding), len(query_embedding))

    print(np.dot(text_embedding, query_embedding))

    # custom_embed_model = Custom_Conch_Text_Embeddimg()
    # embedding = custom_embed_model._get_text_embedding("12314")
    # print(type(embedding), len(embedding), embedding)

    # embed_model = Custom_Conch_Image_Embedding()
    # img_path = "/home/mdi/suri/MDI_RAG_project/data/Image2Image_Basic_Retrieval/med_2class/other_6.png"
    # feature_emb = embed_model.get_image_embedding(img_path)
    # print(type(embedding), len(embedding), embedding)

    # embed_model = Custom_UNI_Image_Embedding()
    # img_path = "/home/mdi/suri/MDI_RAG_project/data/Image2Image_Basic_Retrieval/med_2class/other_6.png"
    # feature_emb = embed_model.get_image_embedding(img_path)
    # print(type(embedding), len(embedding), embedding)