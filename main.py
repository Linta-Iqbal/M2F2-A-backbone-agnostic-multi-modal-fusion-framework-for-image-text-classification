import os
import warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow/XLA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress specific PyTorch warnings if needed
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaModel
import timm

from config.config import Config
from data.dataset import MultimodalNewsDataset
from models.multimodal_model import MultimodalModel
from training.trainer import train_epoch, evaluate
from training.optimizer import create_optimizer_scheduler
from utils.helpers import set_seed, save_data, make_subset, plot_training_metrics, plot_test_metrics

def main():
    config = Config()
    set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    print(f"Training version: {config.TRAINING_VERSION}")
    
    # Create models directory
    models_path = os.path.join(config.MODELS_PATH, config.TRAINING_VERSION)
    if not os.path.isdir(models_path):
        os.makedirs(models_path)
    print(f"Models will be saved to: {models_path}")
    
    # Initialize models
    print("Initializing image model...")
    image_model = timm.create_model(
        'deit_base_distilled_patch16_224',
        pretrained=False,
        num_classes=config.NUM_CLASSES,
        drop_rate=0.3,
        drop_path_rate=0.1
    ).to(device)
    
    print("Initializing text model...")
    text_model = RobertaModel.from_pretrained(config.MODEL_NAME).to(device)
    tokenizer = RobertaTokenizer.from_pretrained(config.MODEL_NAME)
    
    # Create transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = MultimodalNewsDataset(
        config.TEXT_TRAIN_PATH, 
        config.IMG_TRAIN_PATH, 
        tokenizer, 
        config.MAX_LENGTH, 
        transform, 
        config.TEXT_TYPE
    )
    val_dataset = MultimodalNewsDataset(
        config.TEXT_VAL_PATH, 
        config.IMG_VAL_PATH, 
        tokenizer, 
        config.MAX_LENGTH, 
        transform, 
        config.TEXT_TYPE
    )
    test_dataset = MultimodalNewsDataset(
        config.TEXT_TEST_PATH, 
        config.IMG_TEST_PATH, 
        tokenizer, 
        config.MAX_LENGTH, 
        transform, 
        config.TEXT_TYPE
    )
    
    # Apply subset splits if specified in config
    if hasattr(config, 'Training_split') and config.Training_split < 1:
        print(f"Using subset: {config.Training_split * 100}% of data")
        train_dataset = make_subset(train_dataset, config.Training_split, config.SEED)
        val_dataset = make_subset(val_dataset, config.Validation_split, config.SEED)
        test_dataset = make_subset(test_dataset, config.Test_split, config.SEED)
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create dataloaders directly (instead of using create_dataloaders)
    print("Creating dataloaders...")
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Initialize multimodal model
    print("Initializing multimodal model...")
    model = MultimodalModel(text_model, image_model, config.NUM_CLASSES, config).to(device)
    
    # Create optimizer and scheduler
    print("Setting up optimizer and scheduler...")
    optimizer, scheduler = create_optimizer_scheduler(model, config, len(train_loader))
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler() if device.type == 'cuda' else None
    
    # Training state
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    best_test_acc = float('-inf')
    
    # History tracking
    train_history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [],
        'test_loss': [], 'test_acc': [], 'test_f1': [],
        'epochs': []
    }
    
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)
    
    for epoch in range(config.EPOCHS):
        print(f"\n{'='*40}")
        print(f"Epoch {epoch + 1}/{config.EPOCHS}")
        print(f"{'='*40}")
        
        # Train
        print("Training...")
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, scheduler, scaler, device)
        
        # Validate
        print("Validating...")
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # Update history
        train_history['train_loss'].append(train_metrics['loss'])
        train_history['train_acc'].append(train_metrics['accuracy'])
        train_history['train_f1'].append(train_metrics['f1'])
        
        train_history['val_loss'].append(val_metrics['loss'])
        train_history['val_acc'].append(val_metrics['accuracy'])
        train_history['val_f1'].append(val_metrics['f1'])
        
        train_history['epochs'].append(epoch + 1)
        
        # Print metrics
        print(f"\nTraining Metrics:")
        print(f"  Loss: {train_metrics['loss']:.4f}")
        print(f"  Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"  F1 Score: {train_metrics['f1']:.4f}")
        
        print(f"\nValidation Metrics:")
        print(f"  Loss: {val_metrics['loss']:.4f}")
        print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  F1 Score: {val_metrics['f1']:.4f}")
        
        # Check for best validation loss
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_val_acc = val_metrics['accuracy']
            patience_counter = 0
            
            # Save best model
            checkpoint_path = os.path.join(models_path, "multimodal_best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_acc': best_val_acc,
                'config': config.__dict__
            }, checkpoint_path)
            print(f"✓ Saved best model (val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f})")
        else:
            patience_counter += 1
            print(f"✗ No improvement in validation loss for {patience_counter} epochs")
            
            if patience_counter >= config.PATIENCE:
                print(f"\nEarly stopping triggered after {patience_counter} epochs without improvement")
                break
        
        # Test every 2 epochs (as in original code)
        if (epoch + 1) % 2 == 0:
            print("\nTesting on test set...")
            test_metrics = evaluate(model, test_loader, criterion, device)
            
            train_history['test_loss'].append(test_metrics['loss'])
            train_history['test_acc'].append(test_metrics['accuracy'])
            train_history['test_f1'].append(test_metrics['f1'])
            
            print(f"Test Metrics:")
            print(f"  Loss: {test_metrics['loss']:.4f}")
            print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
            print(f"  F1 Score: {test_metrics['f1']:.4f}")
            
            # Save best test model
            if test_metrics['accuracy'] > best_test_acc:
                best_test_acc = test_metrics['accuracy']
                test_checkpoint_path = os.path.join(models_path, "multimodal_test_best_model.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'test_acc': best_test_acc,
                    'test_loss': test_metrics['loss'],
                    'config': config.__dict__
                }, test_checkpoint_path)
                print(f"✓ Saved best test model (acc={best_test_acc:.4f})")
    
    # Final test after training
    print("\n" + "="*50)
    print("Final evaluation on test set...")
    print("="*50)
    
    test_metrics = evaluate(model, test_loader, criterion, device)
    
    # Update history with final test results
    train_history['test_loss'].append(test_metrics['loss'])
    train_history['test_acc'].append(test_metrics['accuracy'])
    train_history['test_f1'].append(test_metrics['f1'])
    
    print(f"\nFinal Test Results:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  F1 Score: {test_metrics['f1']:.4f}")
    
    # Save final model
    final_checkpoint_path = os.path.join(models_path, "multimodal_final_model.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_acc': test_metrics['accuracy'],
        'test_loss': test_metrics['loss'],
        'config': config.__dict__,
        'train_history': train_history
    }, final_checkpoint_path)
    print(f"✓ Saved final model")
    
    # Save configuration and results
    print("\nSaving configuration and results...")
    save_data(
        config=config,
        models_path=models_path,
        train_history=train_history,
        test_metrics=test_metrics,
        best_val_loss=best_val_loss,
        best_val_acc=best_val_acc,
        best_test_acc=best_test_acc
    )
    
    # Create plots
    print("\nCreating visualizations...")
    try:
        plot_training_metrics(train_history, models_path)
        plot_test_metrics(train_history, models_path)
        print("✓ Plots saved successfully")
    except Exception as e:
        print(f"✗ Error creating plots: {e}")
    
    print("\n" + "="*50)
    print("Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best test accuracy: {best_test_acc:.4f}")
    print(f"Models saved in: {models_path}")
    print("="*50)

if __name__ == "__main__":
    main()