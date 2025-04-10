import logging
from transformers import ViTModel
from torch import nn

class HfVisionModel(nn.Module):
    def __init__(self, 
                 hf_model_name_or_path: str,
                 ) -> None:
        super().__init__()
        self.trunk = ViTModel.from_pretrained(hf_model_name_or_path)

    def forward(self, x):
        out = self.trunk(x)
        # print(out.last_hidden_state.shape)
        # print(out.pooler_output.shape)
        # exit()
        return out.pooler_output