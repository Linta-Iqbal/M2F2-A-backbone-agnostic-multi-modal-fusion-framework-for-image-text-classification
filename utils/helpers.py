import os
import random
import numpy as np
import torch
from torch.utils.data import random_split
import json
import matplotlib.pyplot as plt

def save_data(config, models_path, train_history, test_metrics, 
              best_val_loss, best_val_acc, best_test_acc):
    """
    Save training configuration and results.
    """
    config_file = os.path.join(models_path, f"{config.TRAINING_VERSION}_config.txt")
    results_file = os.path.join(models_path, f"{config.TRAINING_VERSION}_results.json")
    
    # Save configuration as text file
    with open(config_file, 'w') as f:
        f.write('='*50 + '\n')
        f.write('MODEL CONFIGURATION\n')
        f.write('='*50 + '\n\n')
        
        f.write('TRAINING PARAMETERS:\n')
        f.write('-'*30 + '\n')
        f.write(f'Epochs: {config.EPOCHS}\n')
        f.write(f'Batch size: {config.BATCH_SIZE}\n')
        f.write(f'Patience: {config.PATIENCE}\n')
        f.write(f'Learning rate: {config.LEARNING_RATE}\n')
        f.write(f'Weight decay: {config.WEIGHT_DECAY}\n')
        f.write(f'Seed: {config.SEED}\n\n')
        
        f.write('MODEL PARAMETERS:\n')
        f.write('-'*30 + '\n')
        f.write(f'Text model: {config.MODEL_NAME}\n')
        f.write(f'Image model: {config.IMAGE_MODEL_NAME}\n')
        f.write(f'Number of classes: {config.NUM_CLASSES}\n')
        f.write(f'Text type: {config.TEXT_TYPE}\n')
        f.write(f'Max length: {config.MAX_LENGTH}\n\n')
        
        f.write('MAMBA PARAMETERS:\n')
        f.write('-'*30 + '\n')
        f.write(f'Mamba dropout: {config.MAMBA_DROPOUT}\n')
        f.write(f'Image d_state: {config.IMAGE_D_STATE}\n')
        f.write(f'Image d_conv: {config.IMAGE_D_CONV}\n')
        f.write(f'Image expand: {config.IMAGE_EXPAND}\n')
        f.write(f'Text d_state: {config.TEXT_D_STATE}\n')
        f.write(f'Text d_conv: {config.TEXT_D_CONV}\n')
        f.write(f'Text expand: {config.TEXT_EXPAND}\n\n')
        
        f.write('LEARNING RATES:\n')
        f.write('-'*30 + '\n')
        f.write(f'Image LR: {config.IMAGE_LR}\n')
        f.write(f'Image WD: {config.IMAGE_WD}\n')
        f.write(f'Text LR: {config.TEXT_LR}\n')
        f.write(f'Text WD: {config.TEXT_WD}\n')
        f.write(f'Mamba LR: {config.MAMBA_LR}\n')
        f.write(f'Mamba WD: {config.MAMBA_WD}\n')
        f.write(f'Attention LR: {config.ATTENTION_LR}\n')
        f.write(f'Attention WD: {config.ATTENTION_WD}\n')
        f.write(f'Classification head LR: {config.CLASSIFICATION_HEAD_LR}\n')
        f.write(f'Classification head WD: {config.CLASSIFICATION_HEAD_WD}\n')
        f.write(f'Classification dropout 1: {config.CLASSIFICATION_HEAD_DROPOUT_1}\n')
        f.write(f'Classification dropout 2: {config.CLASSIFICATION_HEAD_DROPOUT_2}\n\n')
        
        f.write('RESULTS:\n')
        f.write('-'*30 + '\n')
        f.write(f'Final test loss: {test_metrics["loss"]:.4f}\n')
        f.write(f'Final test accuracy: {test_metrics["accuracy"]:.4f}\n')
        f.write(f'Final test F1: {test_metrics["f1"]:.4f}\n')
        f.write(f'Best validation loss: {best_val_loss:.4f}\n')
        f.write(f'Best validation accuracy: {best_val_acc:.4f}\n')
        f.write(f'Best test accuracy: {best_test_acc:.4f}\n')
    
    # Save detailed results as JSON
    results = {
        'config': config.__dict__,
        'final_test_metrics': test_metrics,
        'best_metrics': {
            'val_loss': float(best_val_loss),
            'val_acc': float(best_val_acc),
            'test_acc': float(best_test_acc)
        },
        'train_history': {
            k: [float(x) if isinstance(x, (np.floating, float)) else x for x in v]
            for k, v in train_history.items()
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Configuration saved to: {config_file}")
    print(f"Results saved to: {results_file}")

def make_subset(ds, frac, seed):
    """Create a subset of the dataset."""
    total = len(ds)
    keep = int(frac * total)
    
    subset, _ = random_split(
        ds,
        [keep, total - keep],
        generator=torch.Generator().manual_seed(seed)
    )
    
    print(f"Created subset: {keep}/{total} samples ({frac*100:.1f}%)")
    return subset

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    print(f"Random seed set to: {seed}")

def plot_training_metrics(train_history, save_path):
    """
    Plot training and validation metrics.
    """
    epochs = train_history.get('epochs', list(range(1, len(train_history['train_loss']) + 1)))
    
    plt.figure(figsize=(15, 5))
    
    # Loss plot
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, train_history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Accuracy plot
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    plt.plot(epochs, train_history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # F1 Score plot
    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_history['train_f1'], 'b-', label='Train F1', linewidth=2)
    plt.plot(epochs, train_history['val_f1'], 'r-', label='Val F1', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Training and Validation F1 Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_metrics.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_test_metrics(train_history, save_path):
    """
    Plot test metrics over training epochs.
    """
    if 'test_loss' not in train_history or len(train_history['test_loss']) == 0:
        print("No test metrics to plot")
        return
    
    # Create epoch indices for test metrics (assuming test was run every 2 epochs)
    test_epochs = []
    test_interval = 2  # Test every 2 epochs
    for i in range(len(train_history['test_loss'])):
        test_epochs.append((i + 1) * test_interval)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot loss on primary axis
    ax1.plot(test_epochs, train_history['test_loss'], 'r-o', label='Test Loss', linewidth=2, markersize=8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='r')
    ax1.tick_params(axis='y', labelcolor='r')
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy and F1 on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(test_epochs, train_history['test_acc'], 'g-s', label='Test Accuracy', linewidth=2, markersize=8)
    ax2.plot(test_epochs, train_history['test_f1'], 'b-^', label='Test F1 Score', linewidth=2, markersize=8)
    ax2.set_ylabel('Accuracy / F1 Score', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    
    plt.title('Test Metrics over Training Epochs')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'test_metrics.png'), dpi=300, bbox_inches='tight')
    plt.show()