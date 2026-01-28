import json
import os
from typing import List, Dict, Any
from pathlib import Path

class DataPreprocessor:
    """Utility class for preprocessing and validating multimodal data."""
    
    @staticmethod
    def validate_json_structure(json_file: str, required_fields: List[str] = None) -> bool:
        """
        Validate JSON file structure.
        
        Args:
            json_file: Path to JSON file
            required_fields: List of required fields in each entry
            
        Returns:
            bool: True if valid, False otherwise
        """
        if required_fields is None:
            required_fields = ['headline', 'image_id', 'section']
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                print(f"Error: JSON data should be a list, got {type(data)}")
                return False
            
            for i, item in enumerate(data):
                for field in required_fields:
                    if field not in item:
                        print(f"Error: Missing field '{field}' at index {i}")
                        return False
                    
                    if field == 'image_id' and not isinstance(item[field], (int, str)):
                        print(f"Error: image_id should be int or str at index {i}")
                        return False
                    
                    if field == 'section' and not isinstance(item[field], str):
                        print(f"Error: section should be str at index {i}")
                        return False
            
            print(f"JSON validation passed: {len(data)} valid entries")
            return True
            
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error loading JSON: {e}")
            return False
    
    @staticmethod
    def check_image_files(json_file: str, images_root: str) -> Dict[str, List[str]]:
        """
        Check which image files exist and which are missing.
        
        Args:
            json_file: Path to JSON file
            images_root: Directory containing images
            
        Returns:
            Dict with 'existing' and 'missing' lists
        """
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        existing = []
        missing = []
        
        for item in data:
            image_id = item['image_id']
            image_path = os.path.join(images_root, f"{image_id}.jpg")
            
            if os.path.exists(image_path):
                existing.append(image_path)
            else:
                missing.append(image_path)
        
        print(f"Images: {len(existing)} exist, {len(missing)} missing")
        return {'existing': existing, 'missing': missing}
    
    @staticmethod
    def create_label_mapping(json_file: str, output_file: str = None) -> Dict[str, int]:
        """
        Create and save label mapping.
        
        Args:
            json_file: Path to JSON file
            output_file: Optional path to save mapping
            
        Returns:
            Dictionary mapping label names to IDs
        """
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        labels = set(item['section'] for item in data)
        label2id = {label: i for i, label in enumerate(sorted(labels))}
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'label2id': label2id,
                    'id2label': {i: l for l, i in label2id.items()}
                }, f, indent=2)
            print(f"Label mapping saved to {output_file}")
        
        return label2id
    
    @staticmethod
    def create_train_val_split(json_file: str, train_ratio: float = 0.8, 
                               val_ratio: float = 0.1, test_ratio: float = 0.1,
                               output_dir: str = None, seed: int = 42):
        """
        Create train/val/test splits and save to separate JSON files.
        
        Args:
            json_file: Path to source JSON file
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            output_dir: Directory to save split files
            seed: Random seed for reproducibility
        """
        import random
        random.seed(seed)
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Shuffle data
        random.shuffle(data)
        
        total = len(data)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]
        
        print(f"Splits: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            splits = {
                'train': train_data,
                'val': val_data,
                'test': test_data
            }
            
            for split_name, split_data in splits.items():
                output_path = os.path.join(output_dir, f"{split_name}.json")
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(split_data, f, indent=2)
                print(f"Saved {split_name} split to {output_path}")
        
        return train_data, val_data, test_data