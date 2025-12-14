import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from .config import DEVICE, RESULTS_DIR
from .model import get_resnet18_cifar10, get_dataloaders
from .attacks import pgd_attack
from .interpretability import compute_gradcam, compute_ig, compute_shap, get_explanation_maps
from .metrics import pair_metrics
from .utils import set_seed, denormalize_cifar


def save_saliency_comparison(model_std, model_adv, test_loader, eps=0.1, save_dir=None):
    """
    Generate and save a side-by-side comparison of saliency maps.
    Shows: Clean image, Std GradCAM, Adv GradCAM for both clean and attacked images.
    """
    if save_dir is None:
        save_dir = RESULTS_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    # Get a batch of images
    x, y = next(iter(test_loader))
    x, y = x[:4].to(DEVICE), y[:4].to(DEVICE)  # Use first 4 images
    
    # Generate adversarial examples
    x_adv = pgd_attack(model_std, x, y, eps=eps, alpha=2/255, steps=10)
    
    # Generate saliency maps for clean images
    std_clean_cam = get_explanation_maps(model_std, x, "gradcam")
    adv_clean_cam = get_explanation_maps(model_adv, x, "gradcam")
    
    # Generate saliency maps for attacked images
    std_adv_cam = get_explanation_maps(model_std, x_adv, "gradcam")
    adv_adv_cam = get_explanation_maps(model_adv, x_adv, "gradcam")
    
    # Denormalize images for display
    x_clean_disp = denormalize_cifar(x).cpu().numpy().transpose(0, 2, 3, 1)
    x_adv_disp = denormalize_cifar(x_adv).cpu().numpy().transpose(0, 2, 3, 1)
    
    # Create figure: 4 rows (images) x 6 columns (clean, std_clean, adv_clean, attacked, std_adv, adv_adv)
    fig, axes = plt.subplots(4, 6, figsize=(18, 12))
    
    titles = ['Clean Image', 'Std Model\n(Clean)', 'Adv Model\n(Clean)',
              'Attacked Image', 'Std Model\n(Attacked)', 'Adv Model\n(Attacked)']
    
    for i in range(4):
        # Clean image
        axes[i, 0].imshow(x_clean_disp[i])
        axes[i, 0].axis('off')
        
        # Standard model on clean
        axes[i, 1].imshow(x_clean_disp[i])
        axes[i, 1].imshow(std_clean_cam[i], cmap='jet', alpha=0.5)
        axes[i, 1].axis('off')
        
        # Adversarial model on clean
        axes[i, 2].imshow(x_clean_disp[i])
        axes[i, 2].imshow(adv_clean_cam[i], cmap='jet', alpha=0.5)
        axes[i, 2].axis('off')
        
        # Attacked image
        axes[i, 3].imshow(x_adv_disp[i])
        axes[i, 3].axis('off')
        
        # Standard model on attacked
        axes[i, 4].imshow(x_adv_disp[i])
        axes[i, 4].imshow(std_adv_cam[i], cmap='jet', alpha=0.5)
        axes[i, 4].axis('off')
        
        # Adversarial model on attacked
        axes[i, 5].imshow(x_adv_disp[i])
        axes[i, 5].imshow(adv_adv_cam[i], cmap='jet', alpha=0.5)
        axes[i, 5].axis('off')
    
    # Set titles
    for j, title in enumerate(titles):
        axes[0, j].set_title(title, fontsize=12)
    
    plt.suptitle(f'Saliency Map Comparison (eps={eps}): Standard vs Adversarially Trained', fontsize=14)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'saliency_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saliency comparison saved to {save_path}")
    return save_path

def evaluate_robustness(model, loader, eps_list, attack_steps=10):
    model.eval()
    accuracies = []
    
    print(f"Evaluating robustness...")
    for eps in eps_list:
        correct = 0
        total = 0
        
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            if eps == 0:
                x_adv = x
            else:
                x_adv = pgd_attack(model, x, y, eps, alpha=2/255, steps=attack_steps)
            
            with torch.no_grad():
                logits = model(x_adv)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
                
            if total >= 500:  # Evaluate on 500 samples for better accuracy
                break
                
        acc = correct / total
        accuracies.append(acc)
        print(f"Eps={eps:.4f} -> Acc={acc*100:.2f}%")
        
    return accuracies

def main():
    set_seed(42)
    _, test_loader = get_dataloaders(batch_size=32)
    # Extended range up to 0.20 for comprehensive comparison
    eps_list = [0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20]
    
    # 1. Load Standard Model
    print("\n--- Loading Standard Model ---")
    std_ckpt = "models/resnet18_cifar10.pth"
    model_std = get_resnet18_cifar10(checkpoint_path=std_ckpt)
    acc_std = evaluate_robustness(model_std, test_loader, eps_list)
    
    # 2. Load Adversarial Model
    print("\n--- Loading Adversarial Model ---")
    adv_ckpt = "models/resnet18_cifar10_adv.pth"
    if not os.path.exists(adv_ckpt):
        print("Adversarial model not found! Please run train_adv.py first.")
        return
        
    model_adv = get_resnet18_cifar10(checkpoint_path=adv_ckpt)
    acc_adv = evaluate_robustness(model_adv, test_loader, eps_list)
    
    # 3. Plot Comparison
    plt.figure(figsize=(8, 6))
    plt.plot(eps_list, acc_std, marker='o', label='Standard Training', color='red', linestyle='--')
    plt.plot(eps_list, acc_adv, marker='o', label='Adversarial Training', color='blue')
    
    plt.xlabel("Perturbation Size (Epsilon)")
    plt.ylabel("Accuracy")
    plt.title("Robustness Comparison: Standard vs Adversarial Training")
    plt.legend()
    plt.grid(True)
    plt.ylim(-0.05, 1.05)
    
    save_path = os.path.join(RESULTS_DIR, "robustness_comparison.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nComparison plot saved to {save_path}")
    
    # 4. Generate Saliency Map Comparison
    print("\n--- Generating Saliency Map Comparison ---")
    save_saliency_comparison(model_std, model_adv, test_loader, eps=0.1)

if __name__ == "__main__":
    main()
