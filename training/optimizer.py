import torch
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

def create_optimizer_scheduler(model, config, train_loader_length):
    """
    Create optimizer with different learning rates for different components.
    
    Args:
        model: The multimodal model
        config: Configuration object
        train_loader_length: Length of training dataloader
    
    Returns:
        optimizer, scheduler
    """
    # Create a set to track which parameters have been added
    added_params = set()
    optimizer_params = []
    
    # Function to add parameters if they haven't been added yet
    def add_params(param_list, lr, wd, name=""):
        params_to_add = []
        for p in param_list:
            if id(p) not in added_params and p.requires_grad:
                params_to_add.append(p)
                added_params.add(id(p))
        
        if params_to_add:
            optimizer_params.append({
                'params': params_to_add,
                'lr': lr,
                'weight_decay': wd
            })
            print(f"  {name}: Added {len(params_to_add)} parameters with lr={lr}, wd={wd}")
        else:
            print(f"  {name}: No parameters to add")
    
    print("\nSetting up optimizer parameter groups:")
    
    # 1. Image model parameters
    image_params = [p for n, p in model.image_model.named_parameters() if p.requires_grad]
    add_params(image_params, config.IMAGE_LR, config.IMAGE_WD, "Image model")
    
    # 2. Text model parameters
    text_params = [p for n, p in model.text_model.named_parameters() if p.requires_grad]
    add_params(text_params, config.TEXT_LR, config.TEXT_WD, "Text model")
    
    # 3. Mamba blocks (aligner)
    mamba_params = [p for n, p in model.named_parameters() 
                   if 'aligner' in n and p.requires_grad]
    add_params(mamba_params, config.MAMBA_LR, config.MAMBA_WD, "Mamba blocks")
    
    # 4. Attention modules
    attention_params = [p for n, p in model.named_parameters() 
                       if 'attention' in n and p.requires_grad]
    add_params(attention_params, config.ATTENTION_LR, config.ATTENTION_WD, "Attention modules")
    
    # 5. Classification heads
    classification_params = [p for n, p in model.named_parameters() 
                           if 'multiclassification' in n and p.requires_grad]
    add_params(classification_params, config.CLASSIFICATION_HEAD_LR, 
               config.CLASSIFICATION_HEAD_WD, "Classification heads")
    
    # 6. Any remaining parameters (default settings)
    all_params = set(model.parameters())
    remaining_params = []
    for p in all_params:
        if id(p) not in added_params and p.requires_grad:
            remaining_params.append(p)
    
    if remaining_params:
        optimizer_params.append({
            'params': remaining_params,
            'lr': config.LEARNING_RATE,
            'weight_decay': config.WEIGHT_DECAY
        })
        print(f"  Default: Added {len(remaining_params)} remaining parameters with lr={config.LEARNING_RATE}, wd={config.WEIGHT_DECAY}")
    
    # Create optimizer
    optimizer = AdamW(optimizer_params)
    
    # Count total trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {total_params:,}")
    
    # Create scheduler
    total_steps = train_loader_length * config.EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.15 * total_steps),
        num_training_steps=total_steps
    )
    
    print(f"Scheduler: Cosine with {int(0.15 * total_steps)} warmup steps, {total_steps} total steps")
    
    return optimizer, scheduler