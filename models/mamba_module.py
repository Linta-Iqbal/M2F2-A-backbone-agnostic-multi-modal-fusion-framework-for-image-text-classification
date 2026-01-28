import torch
import torch.nn as nn
from mamba_ssm import Mamba

class DualModalityMamba(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.unified_mamba = Mamba(
            d_model=768, 
            d_state=config.IMAGE_D_STATE, 
            d_conv=config.IMAGE_D_CONV, 
            expand=config.IMAGE_EXPAND
        )
        
        self.dropout = nn.Dropout(config.MAMBA_DROPOUT)
        self.layer_norm = nn.LayerNorm(768)
        
    def forward(self, text_features, image_features):
        text_seq_len = text_features.size(1)
        
        combined_features = torch.cat([text_features, image_features], dim=1)
        combined_out = self.unified_mamba(combined_features) + combined_features
        combined_out = self.layer_norm(combined_out)
        
        text_out = combined_out[:, :text_seq_len, :]
        image_out = combined_out[:, text_seq_len:, :]

        text_out = text_out[:, 0, :]
        image_out = (image_out[:, 0] + image_out[:, 1]) / 2
        
        return text_out, image_out