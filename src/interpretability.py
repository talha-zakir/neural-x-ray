import torch
import torch.nn as nn
import numpy as np
from captum.attr import IntegratedGradients, GradientShap
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from .config import DEVICE
from .metrics import map_to_heatmap

def get_last_conv_layer(model):
    """
    Return the last Conv2d module (for ResNet-18 this will be layer4[-1].conv2).
    """
    last_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    return last_conv

def compute_gradcam(model, x, target_class=None):
    """
    x: normalized image tensor [B,3,H,W]
    returns: numpy array [B,H,W] normalized to [0,1]
    """
    target_layer = get_last_conv_layer(model)
    gradcam = GradCAM(
        model=model,
        target_layers=[target_layer],
    )

    x = x.to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        preds = logits.argmax(dim=1)

    targets = []
    for i in range(x.size(0)):
        cls = int(preds[i].item()) if target_class is None else int(target_class)
        targets.append(ClassifierOutputTarget(cls))

    # GradCAM returns [B, H, W]
    grayscale_cam = gradcam(input_tensor=x, targets=targets)
    cams = np.array([map_to_heatmap(cam) for cam in grayscale_cam])
    return cams

def compute_ig(model, x, target_class=None, n_steps=50):
    """
    x: normalized tensor [B,3,H,W]
    returns: numpy array [B,H,W] aggregated over channels
    """
    ig = IntegratedGradients(model)
    x = x.to(DEVICE).clone().detach()
    x.requires_grad_(True)

    baseline = torch.zeros_like(x)

    with torch.no_grad():
        logits = model(x)
        preds = logits.argmax(dim=1)

    if target_class is None:
        target = preds
    else:
        target = torch.full_like(preds, int(target_class))

    attributions = ig.attribute(
        x,
        baselines=baseline,
        target=target,
        n_steps=n_steps,
    )  # [B,3,H,W]

    # Aggregate over channels
    attr = attributions.abs().sum(dim=1)  # [B,H,W]
    attr_np = attr.detach().cpu().numpy()
    attr_np = np.array([map_to_heatmap(a) for a in attr_np])
    return attr_np

def compute_shap(model, x, target_class=None, n_baseline=50, noise_std=0.1):
    """
    Approximates SHAP using GradientShap.
    x: normalized tensor [B,3,H,W]
    returns: numpy array [B,H,W]
    """
    gs = GradientShap(model)
    x = x.to(DEVICE).clone().detach()
    x.requires_grad_(True)

    # baseline distribution around zero
    baseline_dist = torch.randn(
        (n_baseline,) + x.shape[1:], device=DEVICE
    ) * noise_std

    with torch.no_grad():
        logits = model(x)
        preds = logits.argmax(dim=1)

    if target_class is None:
        target = preds
    else:
        target = torch.full_like(preds, int(target_class))

    # GradientShap
    attributions = gs.attribute(
        x,
        baselines=baseline_dist,
        target=target,
        n_samples=n_baseline,
    )  # [B,3,H,W]

    attr = attributions.abs().sum(dim=1)  # [B,H,W]
    attr_np = attr.detach().cpu().numpy()
    attr_np = np.array([map_to_heatmap(a) for a in attr_np])
    return attr_np

def get_explanation_maps(model, x, method: str):
    """
    x: normalized tensor [B,3,H,W]
    method: 'gradcam' | 'ig' | 'shap'
    returns: numpy [B,H,W]
    """
    method = method.lower()
    if method == "gradcam":
        return compute_gradcam(model, x)
    elif method == "ig":
        return compute_ig(model, x)
    elif method == "shap":
        return compute_shap(model, x)
    else:
        raise ValueError(f"Unknown method: {method}")
