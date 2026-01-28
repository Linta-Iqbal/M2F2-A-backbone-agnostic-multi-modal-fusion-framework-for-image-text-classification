import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

def train_epoch(model, train_loader, optimizer, criterion, scheduler, scaler, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for text_inputs, images, labels in tqdm(train_loader):
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(text_inputs, images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1': f1
    }

def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for text_inputs, images, labels in tqdm(test_loader):
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(text_inputs, images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1': f1
    }