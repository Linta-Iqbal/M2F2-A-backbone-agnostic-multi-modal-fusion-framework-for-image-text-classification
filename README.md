M2F2: A backbone-agnostic multi-modal fusion framework for image-text classification

We proposed a backbone-agnostic multimodal framework for modeling inter and intra-modal interactions in a data-driven way for image-text classification.

## 🏗️ Project Structure
```text
M2F2_Project/
│
├── config/                 # Configuration management
│   ├── __init__.py
│   └── config.py           # All hyperparameters and paths
│
├── data/                   # Data handling
│   ├── __init__.py
│   ├── dataset.py          # Multimodal dataset class
│   ├── dataloader.py          # Multimodal dataloader class
│   └── data_preprocessor.py # Data preprocessing
│
├── models/                 # Model architectures
│   ├── __init__.py
│   ├── mamba_module.py     # MSM
│   ├── attention_module.py # IntRel Cap
│   ├── classification.py  # MFC
│   ├── post_layer.py      # Post-processing layer
│   └── multimodal_model.py # Main multimodal model
│
├── training/               # Training utilities
│   ├── __init__.py
│   ├── trainer.py          # Training and evaluation functions
│   └── optimizer.py        # Optimizer and scheduler setup
│
├── utils/                  # Helper functions
│   ├── __init__.py
│   └── helpers.py          # Utilities, metrics, plotting
│
├── main.py                 # Main training script
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```


## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for training)

### Installation
1. Clone the repository
2. Create a virtual environment
3. Install dependencies using `requirements.txt`

## ⚙️ Configuration Steps

1. **Set Dataset Paths**: Open `config/config.py` and update the dataset and model paths.
2. **Run the Project**: `python main.py`
3. **Change Text Type**: To use different text fields from your dataset (e.g., caption, headline, abstract):
   - Open `config/config.py`
   - Modify the `TEXT_TYPE` variable

## 📊 Outputs Generated

### Saved Models
- `multimodal_best_model.pt`: Best model based on validation loss
- `multimodal_test_best_model.pt`: Best model based on test accuracy  
- `multimodal_final_model.pt`: Final trained model

### Configuration and Results
- `{TRAINING_VERSION}_config.txt`: Training configuration
- `{TRAINING_VERSION}_results.json`: Detailed training metrics

### Visualizations
- `training_metrics.png`: Training/validation loss, accuracy, F1 scores
- `test_metrics.png`: Test metrics over training epochs

## 🔧 Using a Different Dataset
1. Set the dataset paths in `config/config.py`
2. Update `NUM_CLASSES` to match your dataset
3. Use the respective dataset split from the `data/data_splits/` folder.
4. Modify other relevant configuration parameters.

## Datasets
1. MM-IMDB: Gated Multimodal Units for Information Fusion (http://arxiv.org/abs/1702.01992)
2. MS-COCO 2017: Microsoft COCO: Common Objects in Context (https://link.springer.com/chapter/10.1007/978-3-319-10602-1_48)
3. Food101: Recipe recognition with large multimodal food dataset (https://ieeexplore.ieee.org/document/7169757/)
4. N24News: A New Dataset for Multimodal News Classification (https://aclanthology.org/2022.lrec-1.729/)

## Baselines
1. EF (https://ieeexplore.ieee.org/document/8269806)
2. GMU (http://arxiv.org/abs/1702.01992)
3. UniS:MMC (https://aclanthology.org/2023.findings-acl.41/)
4. Dynamic Fusion (https://ieeexplore.ieee.org/document/10208451/)
5. MMPareto (https://proceedings.mlr.press/v235/wei24d.html)
6. ReconBoost (https://proceedings.mlr.press/v235/hua24a.html)
7. I2MoE-MulT (https://proceedings.mlr.press/v267/xin25c.html)
8. UDML (https://openaccess.thecvf.com/content/CVPR2026F/papers/Wei_Unbiased_Dynamic_Multimodal_Fusion_CVPRF_2026_paper.pdf)




