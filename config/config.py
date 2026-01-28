import os
from dataclasses import dataclass

@dataclass
class Config:
    # Model Names
    MODEL_NAME = "roberta-base"
    IMAGE_MODEL_NAME = "DeiT"
    
    # Training
    SEED = 42
    BATCH_SIZE = 16
    EPOCHS = 40
    PATIENCE = 10
    NUM_CLASSES = 24
    
    # Text
    MAX_LENGTH = 256
    TEXT_TYPE = 'headline'
    
    # Learning Rates
    LEARNING_RATE = 4e-5
    WEIGHT_DECAY = 0.03
    IMAGE_LR = 4e-5
    IMAGE_WD = 0.05
    TEXT_LR = 2e-5
    TEXT_WD = 1e-4
    
    # Mamba Parameters
    MAMBA_DROPOUT = 0.1
    MAMBA_LR = 4e-5
    MAMBA_WD = 0.03
    
    # Attention
    ATTENTION_LR = 4e-5
    ATTENTION_WD = 0.03
    
    # Classification Head
    CLASSIFICATION_HEAD_LR = 4e-5
    CLASSIFICATION_HEAD_WD = 0.03
    CLASSIFICATION_HEAD_DROPOUT_1 = 0.2
    CLASSIFICATION_HEAD_DROPOUT_2 = 0.1
    
    # Mamba Architecture
    IMAGE_D_STATE = 16
    IMAGE_D_CONV = 4
    IMAGE_EXPAND = 2
    TEXT_D_STATE = 16
    TEXT_D_CONV = 4
    TEXT_EXPAND = 2
    
    # Subset splits (if using subsets)
    Training_split = 1.0    # Use full dataset
    Validation_split = 1.0  # Use full dataset
    Test_split = 1.0        # Use full dataset
    
    # Paths
    IMG_TRAIN_PATH = '/mnt/e/FINAL DATASETS/N24News/split_images/train'
    IMG_VAL_PATH = '/mnt/e/FINAL DATASETS/N24News/split_images/val'
    IMG_TEST_PATH = '/mnt/e/FINAL DATASETS/N24News/split_images/test'
    
    TEXT_TRAIN_PATH = "/mnt/e/FINAL DATASETS/N24News/news/nytimes_train.json"
    TEXT_VAL_PATH = "/mnt/e/FINAL DATASETS/N24News/news/nytimes_dev.json"
    TEXT_TEST_PATH = "/mnt/e/FINAL DATASETS/N24News/news/nytimes_test.json"
    
    MODELS_PATH = "/mnt/e/PC2_Saved_Models/N24News/"
    TRAINING_VERSION = "Headline"
    
    # Additional paths for saving
    @property
    def model_save_path(self):
        return os.path.join(self.MODELS_PATH, self.TRAINING_VERSION)