import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from transformers import RobertaTokenizer

class MultimodalNewsDataset(Dataset):
    def __init__(self, json_file, images_root, tokenizer, max_length=256, transform=None, text_type='headline'):
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.images_root = images_root
        self.transform = transform
        self.text_type = text_type

        self.texts = [item[text_type] for item in self.data]
        self.image_ids = [item['image_id'] for item in self.data]
        self.raw_labels = [item['section'] for item in self.data]

        self.unique_labels = sorted(set(self.raw_labels))
        self.label2id = {lab: i for i, lab in enumerate(self.unique_labels)}
        self.labels = [self.label2id[lab] for lab in self.raw_labels]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Text
        text = self.texts[idx]
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length, 
                            padding='max_length', return_tensors='pt')
        text_inputs = {k: v.squeeze(0) for k, v in enc.items()}

        # Image
        image_path = os.path.join(self.images_root, f"{self.image_ids[idx]}.jpg")
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        # Label
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return text_inputs, image, label