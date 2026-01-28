import torch
import torch.nn as nn
import torch.nn.functional as F
from .post_layer import PostLayer

class MultiClassificationHead(nn.Module):
    def __init__(self, num_labels, config):
        super().__init__()
        self.config = config
        
        self.classification_head1 = nn.Sequential(
            nn.Linear(768, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(config.CLASSIFICATION_HEAD_DROPOUT_1),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(config.CLASSIFICATION_HEAD_DROPOUT_2),
            nn.Linear(512, num_labels)
        )

        self.classification_head2 = nn.Sequential(
            nn.Linear(768, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(config.CLASSIFICATION_HEAD_DROPOUT_1),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(config.CLASSIFICATION_HEAD_DROPOUT_2),
            nn.Linear(512, num_labels)
        )

        self.classification_head3 = nn.Sequential(
            nn.Linear(1536, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(config.CLASSIFICATION_HEAD_DROPOUT_1),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(config.CLASSIFICATION_HEAD_DROPOUT_2),
            nn.Linear(512, num_labels)
        )
        self.post_layer = PostLayer(config)

    def forward(self, text_features, image_features):
        multiply = text_features * image_features
        add = text_features + image_features
        concat = torch.cat([text_features, image_features], dim=-1)
        
        multiply, add, concat = self.post_layer(multiply, add, concat)
        
        out_multiply = self.classification_head1(multiply)
        out_add = self.classification_head2(add)
        out_concat = self.classification_head3(concat)

        probs_multiply = F.softmax(out_multiply, dim=1)
        probs_add = F.softmax(out_add, dim=1)
        probs_concat = F.softmax(out_concat, dim=1)

        sorted_probs_multiply, sorted_indices_multiply = torch.sort(probs_multiply, dim=1, descending=True)
        sorted_probs_add, sorted_indices_add = torch.sort(probs_add, dim=1, descending=True)
        sorted_probs_concat, sorted_indices_concat = torch.sort(probs_concat, dim=1, descending=True)

        final_prob = []
        for i in range(text_features.size(0)):
            m1 = sorted_indices_multiply[i, 0].item()
            a1 = sorted_indices_add[i, 0].item()
            c1 = sorted_indices_concat[i, 0].item()

            if m1 == a1 == c1:
                winner = "multiply"
            elif a1 == m1:
                winner = 'add'
            elif c1 == a1:
                winner = 'concat'
            elif m1 == c1:
                winner = 'multiply'
            else:
                m_top = sorted_probs_multiply[i].std()
                a_top = sorted_probs_add[i].std()
                c_top = sorted_probs_concat[i].std()
                spread_dict = {
                    m_top: "multiply",
                    a_top: "add",
                    c_top: "concat"
                }
                winner = spread_dict[max(spread_dict.keys())]

            if winner == "multiply":
                final_prob.append(out_multiply[i])
            elif winner == "add":
                final_prob.append(out_add[i])
            elif winner == "concat":
                final_prob.append(out_concat[i])

        return torch.stack(final_prob, dim=0)