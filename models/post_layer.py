import torch.nn as nn
from mamba_ssm import Mamba

class PostLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.cross_mamba_multiply = Mamba(
            d_model=768, 
            d_state=config.IMAGE_D_STATE, 
            d_conv=config.IMAGE_D_CONV, 
            expand=config.IMAGE_EXPAND
        )
        self.cross_mamba_add = Mamba(
            d_model=768, 
            d_state=config.IMAGE_D_STATE, 
            d_conv=config.IMAGE_D_CONV, 
            expand=config.IMAGE_EXPAND
        )
        self.cross_mamba_concate = Mamba(
            d_model=1536, 
            d_state=config.IMAGE_D_STATE, 
            d_conv=config.IMAGE_D_CONV, 
            expand=config.IMAGE_EXPAND
        )
        
        self.layer_norm_c = nn.LayerNorm(1536) 
        self.layer_norm = nn.LayerNorm(768)
        
    def forward(self, multiply_features, add_features, concat_features):
        multiply_features = self.cross_mamba_multiply(multiply_features)
        multiply_features = self.layer_norm(multiply_features) 
        multiply_features = multiply_features.squeeze(1)

        add_features = self.cross_mamba_add(add_features)
        add_features = self.layer_norm(add_features) 
        add_features = add_features.squeeze(1)

        concat_features = self.cross_mamba_concate(concat_features)
        concat_features = self.layer_norm_c(concat_features)
        concat_features = concat_features.squeeze(1)
        
        return multiply_features, add_features, concat_features