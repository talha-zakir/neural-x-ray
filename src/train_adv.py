import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
from .config import DEVICE, RESULTS_DIR
from .model import get_resnet18_cifar10, get_dataloaders, evaluate_accuracy
from .attacks import pgd_attack

def train_adv_model(epochs=10, batch_size=128, lr=0.01, eps=8/255, alpha=2/255, steps=7):
    """Train a ResNet18 model using adversarial training with PGD attacks."""
    print(f"Adversarial Training on device: {DEVICE}")
    print(f"Attack Params: eps={eps:.4f}, alpha={alpha:.4f}, steps={steps}")
    
    # 1. Prepare Data
    train_loader, test_loader = get_dataloaders(batch_size=batch_size)
    
    # 2. Prepare Model - START FROM PRETRAINED STANDARD MODEL
    std_checkpoint = "models/resnet18_cifar10.pth"
    if os.path.exists(std_checkpoint):
        print(f"Loading pretrained standard model from {std_checkpoint}")
        model = get_resnet18_cifar10(checkpoint_path=std_checkpoint)
    else:
        print("No pretrained model found, starting from ImageNet weights")
        model = get_resnet18_cifar10(checkpoint_path=None)
    model.train()
    
    # 3. Loss & Optimizer - Use SGD with momentum for better convergence
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 4. Training Loop
    best_acc = 0.0
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "resnet18_cifar10_adv.pth")
    
    for epoch in range(epochs):
        start_time = time.time()
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                
            model.eval() # Eval mode for generating attack
            adv_inputs = pgd_attack(model, inputs, labels, eps=eps, alpha=alpha, steps=steps)
            model.train() # Back to train mode
            
            optimizer.zero_grad()
            outputs = model(adv_inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            if (i + 1) % 50 == 0:
                print(f"    Batch [{i+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

        epoch_time = time.time() - start_time
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | Time: {epoch_time:.1f}s")
        
        # Save checkpoint after each epoch
        torch.save(model.state_dict(), save_path)
        print(f"  -> Checkpoint saved to {save_path}")
        
        # Step the learning rate scheduler
        scheduler.step()

    print(f"\nAdversarial Training Complete.")
    return save_path

if __name__ == "__main__":
    train_adv_model(epochs=25)
