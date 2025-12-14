import torch
import torch.nn.functional as F
from torchvision import transforms
from .config import DEVICE, CIFAR_MEAN, CIFAR_STD
from .utils import clamp_01, denormalize_cifar

def fgsm_attack(model, x, y, eps):
    """
    x: normalized tensor [B,3,32,32]
    y: labels
    eps: in [0,1] image scale (e.g. 8/255)
    """
    model.eval()

    x = x.clone().detach().to(DEVICE)
    y = y.to(DEVICE)

    x.requires_grad_(True)

    # Make sure gradients are enabled even if called under no_grad()
    with torch.enable_grad():
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        model.zero_grad()
        loss.backward()
        if x.grad is None:
            grad = torch.zeros_like(x)
        else:
            grad = x.grad.detach()

    # FGSM step in normalized space (but we apply perturbation in 'image' space effectively?)
    # The snippet from notebook:
    # x_adv = x + eps * torch.sign(grad)
    # The note says "FGSM step in normalized space" but then checks bounds?
    # Actually, if we add eps to normalized x directly, eps implies normalized-scale perturbation.
    # But common FGSM is x_adv = x + eps * sign(data_grad).
    # If eps is 8/255, that is small in 0-1 range.
    # If x is normalized (mean=0.5, std=0.2), then 8/255 is ~0.03.
    # 0.03 in normalized space is 0.03 * std?? No.
    # The notebook code: x_adv = x + eps * torch.sign(grad)
    # Then: x_adv = clamp_01(denormalize_cifar(x_adv))
    # This implies x was normalized. 
    # If x is normalized, adding eps directly is strictly correct ONLY if eps is intended for normalized space.
    # Usually eps is defined in pixel space (e.g. 8/255).
    # If we add it to normalized tensor, it might be off scale.
    # BUT, the notebook does: x + eps * sign(grad). sign(grad) is -1 or 1.
    # So we move by eps in each dimension.
    # If eps is 8/255 (~0.03), and x is normalized (approx -2 to 2), moving by 0.03 is fine.
    # Then it calls denote_cifar.
    # Let's stick to the notebook implementation exactly to ensure reproducibility of their method.
    
    x_adv = x + eps * torch.sign(grad)

    # Go back to [0,1] pixel space
    x_adv = clamp_01(denormalize_cifar(x_adv))

    # Re-normalize to feed to the model
    x_adv = transforms.Normalize(CIFAR_MEAN, CIFAR_STD)(x_adv)
    return x_adv.detach()


def pgd_attack(model, x, y, eps, alpha, steps, random_start=True):
    """
    Projected Gradient Descent (L_infinity).
    eps, alpha are in [0,1] image scale.
    """
    model.eval()

    x = x.clone().detach().to(DEVICE)
    y = y.to(DEVICE)

    # Work in image space [0,1]
    x_img = clamp_01(denormalize_cifar(x))

    if random_start:
        x_img = x_img + torch.empty_like(x_img).uniform_(-eps, eps)
        x_img = clamp_01(x_img)

    x_adv = x_img.clone().detach()
    x_adv.requires_grad_(True)

    for _ in range(steps):
        # Enable grads inside the loop
        with torch.enable_grad():
            x_norm = transforms.Normalize(CIFAR_MEAN, CIFAR_STD)(x_adv)
            logits = model(x_norm)
            loss = F.cross_entropy(logits, y)
            model.zero_grad()
            loss.backward()
            if x_adv.grad is None:
                grad = torch.zeros_like(x_adv)
            else:
                grad = x_adv.grad.detach()

        # PGD update (in image space)
        x_adv = x_adv + alpha * torch.sign(grad)

        # Project back to eps-ball around original x_img
        eta = torch.clamp(x_adv - x_img, min=-eps, max=eps)
        x_adv = clamp_01(x_img + eta).detach()
        x_adv.requires_grad_(True)

    # Final normalize for model
    x_adv_norm = transforms.Normalize(CIFAR_MEAN, CIFAR_STD)(x_adv)
    return x_adv_norm.detach()
