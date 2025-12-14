import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
from .config import DEVICE, CIFAR_MEAN, CIFAR_STD, DATA_DIR

def get_resnet18_cifar10(checkpoint_path=None):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Modify for CIFAR-10 (small images 32x32)
    # 1. Replace first 7x7 conv with 3x3 conv, stride 1
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # 2. Remove maxpool (it downsamples too early for 32x32)
    model.maxpool = nn.Identity()
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)

    if checkpoint_path is not None and os.path.isfile(checkpoint_path):
        print("Loading checkpoint from", checkpoint_path)
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state)

    model.to(DEVICE)
    return model

def get_dataloaders(batch_size=128, num_workers=2):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_dataset = datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

@torch.no_grad()
def evaluate_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        logits = model(x)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / total
