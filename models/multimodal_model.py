import torch.nn as nn
import timm
from transformers import RobertaModel
from .mamba_module import DualModalityMamba
from .attention_module import AttentionModule
from .classification import MultiClassificationHead

class MultimodalModel(nn.Module):
    def __init__(self, text_model, image_model, num_labels, config):
        super().__init__()
        self.text_model = text_model
        self.image_model = image_model
        self.num_labels = num_labels
        self.config = config

        self.aligner = DualModalityMamba(config)
        self.attention = AttentionModule(config)
        self.multiclassification = MultiClassificationHead(num_labels, config)

    def forward(self, text_inputs, images):
        text_outputs = self.text_model(**text_inputs)
        text_features = text_outputs.last_hidden_state[:, :, :]

        image_features = self.image_model.forward_features(images)

        text_features_aligned, image_features_aligned = self.aligner(text_features, image_features)
        text_attented_features, image_attented_features = self.attention(text_features_aligned, image_features_aligned)
        output = self.multiclassification(text_attented_features, image_attented_features)

        return output