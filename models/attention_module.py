import torch.nn as nn

class AttentionModule(nn.Module):
    def __init__(self, config, d=768, heads=12, dropout=0.2):
        super().__init__()
        self.config = config

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d, 
            num_heads=heads, 
            dropout=dropout, 
            batch_first=False
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d, 
            num_heads=heads, 
            dropout=dropout, 
            batch_first=False
        )

        self.norm_self = nn.LayerNorm(d)
        self.norm_cross = nn.LayerNorm(d)

    def forward(self, text_features, image_features):
        text_features = text_features.unsqueeze(1)
        image_features = image_features.unsqueeze(1)

        text = text_features.transpose(0, 1)    # [L_text, B, D]
        image = image_features.transpose(0, 1)  # [L_image, B, D]

        # Text Self-Attention + Residual + Norm
        text_sa, _ = self.self_attn(text, text, text)  
        text = self.norm_self(text + text_sa)      # Residual + Norm

        # Image Self-Attention + Residual + Norm
        image_sa, _ = self.self_attn(image, image, image)  
        image = self.norm_self(image + image_sa)
        
        # Cross-Attention + Residual + Norm
        text_ca, _ = self.cross_attn(text, image, image)   # text attends to image
        text_out = self.norm_cross(text + text_ca)    

        # Cross-Attention + Residual + Norm
        image_ca, _ = self.cross_attn(image, text, text)   # image attends to text
        image_out = self.norm_cross(image + image_ca)

        text_out = text_out.transpose(0, 1)
        image_out = image_out.transpose(0, 1)

        return image_out, text_out