import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
from .config import DEVICE, RESULTS_DIR
from .model import get_resnet18_cifar10, get_dataloaders, evaluate_accuracy

def train_model(epochs=5, batch_size=128, lr=0.001):
    print(f"Training on device: {DEVICE}")
    
    # 1. Prepare Data
    train_loader, test_loader = get_dataloaders(batch_size=batch_size)
    
    # 2. Prepare Model
    model = get_resnet18_cifar10(checkpoint_path=None) # Start from ImageNet weights
    model.train()
    
    # 3. Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 4. Training Loop
    best_acc = 0.0
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "resnet18_cifar10.pth")
    
    for epoch in range(epochs):
        start_time = time.time()
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        # Evaluation
        train_acc = evaluate_accuracy(model, train_loader)
        test_acc = evaluate_accuracy(model, test_loader)
        model.train() # Switch back to train mode
        
        duration = time.time() - start_time
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Loss: {running_loss/len(train_loader):.4f} "
              f"Train Acc: {train_acc*100:.2f}% "
              f"Test Acc: {test_acc*100:.2f}% "
              f"Time: {duration:.1f}s")
        
        # Save Best
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)
            print(f"  -> Model saved to {save_path}")

    print(f"\nTraining Complete. Best Test Acc: {best_acc*100:.2f}%")
    return save_path

if __name__ == "__main__":
    train_model(epochs=5)
