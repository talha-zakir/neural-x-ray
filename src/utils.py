import torch
import numpy as np
import random
import os
from .config import CIFAR_MEAN, CIFAR_STD, DEVICE

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def denormalize_cifar(x: torch.Tensor) -> torch.Tensor:
    """
    x: [B, 3, H, W] normalized with CIFAR_MEAN/STD
    returns de-normalized tensor in [0, 1]
    """
    mean = torch.tensor(CIFAR_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD, device=x.device).view(1, 3, 1, 1)
    x = x * std + mean
    return torch.clamp(x, 0.0, 1.0)

def clamp_01(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, 1.0)

def prep_for_model(x: torch.Tensor) -> torch.Tensor:
    """
    Ensure tensor is float and on DEVICE.
    """
    return x.to(DEVICE).float()
