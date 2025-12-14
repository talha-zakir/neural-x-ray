import matplotlib.pyplot as plt
import numpy as np
import os
from .config import RESULTS_DIR
from .utils import denormalize_cifar

def save_saliency_comparison(x_clean, x_adv, map_clean, map_adv, method, attack, eps, image_id):
    """
    Save a comparison plot: Clean Image | Clean Map | Adv Image | Adv Map
    x_clean, x_adv: normalized tensors [C, H, W]
    map_clean, map_adv: saliency maps [H, W] (normalized 0-1)
    """
    # Denormalize images for visualization
    img_clean = denormalize_cifar(x_clean.unsqueeze(0)).squeeze(0).cpu().numpy().transpose(1, 2, 0)
    img_adv = denormalize_cifar(x_adv.unsqueeze(0)).squeeze(0).cpu().numpy().transpose(1, 2, 0)
    
    # Clip to valid range just in case
    img_clean = np.clip(img_clean, 0, 1)
    img_adv = np.clip(img_adv, 0, 1)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Clean Image
    axes[0].imshow(img_clean)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Clean Saliency
    axes[1].imshow(img_clean) # Background
    im1 = axes[1].imshow(map_clean, cmap='jet', alpha=0.5)
    axes[1].set_title(f"{method.upper()} (Clean)")
    axes[1].axis('off')
    
    # Adversarial Image
    axes[2].imshow(img_adv)
    axes[2].set_title(f"Adv Image ({attack.upper()}, eps={eps:.3f})")
    axes[2].axis('off')
    
    # Adversarial Saliency
    axes[3].imshow(img_adv) # Background
    im2 = axes[3].imshow(map_adv, cmap='jet', alpha=0.5)
    axes[3].set_title(f"{method.upper()} (Adv)")
    axes[3].axis('off')
    
    plt.tight_layout()
    
    filename = f"saliency_{method}_{attack}_eps{eps:.3f}_img{image_id}.png"
    save_path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    return save_path

def plot_metrics_vs_eps(results, output_filename="metrics_vs_eps.png"):
    """
    Plot metrics vs epsilon for different methods/attacks.
    results: list of result dictionaries from main execution
    """
    # Parse results
    # We want to plot: SSIM vs Eps, IoU vs Eps for each (Attack, Method) pair
    
    # Group results by (Attack, Method)
    groups = {}
    for res in results:
        key = (res['attack'], res['method'])
        if key not in groups:
            groups[key] = {'eps': [], 'ssim': [], 'iou': [], 'entropy_clean': [], 'entropy_adv': [], 'corr': []}
        
        groups[key]['eps'].append(res['eps'])
        groups[key]['ssim'].append(res['ssim'])
        groups[key]['iou'].append(res['iou'])
        groups[key]['entropy_clean'].append(res.get('entropy_clean', 0))
        groups[key]['entropy_adv'].append(res.get('entropy_adv', 0))
        groups[key]['corr'].append(res.get('corr', 0))
    
    # Sort by epsilon
    for key in groups:
        sorted_indices = np.argsort(groups[key]['eps'])
        for metric in groups[key]:
            groups[key][metric] = np.array(groups[key][metric])[sorted_indices]

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot SSIM
    ax = axes[0]
    for (attack, method), data in groups.items():
        label = f"{method.upper()} ({attack.upper()})"
        ax.plot(data['eps'], data['ssim'], marker='o', label=label)
    ax.set_xlabel("Epsilon")
    ax.set_ylabel("SSIM")
    ax.set_title("Saliency Map Similarity (SSIM) vs Perturbation Size")
    ax.legend()
    ax.grid(True)
    
    # Plot IoU
    ax = axes[1]
    for (attack, method), data in groups.items():
        label = f"{method.upper()} ({attack.upper()})"
        ax.plot(data['eps'], data['iou'], marker='s', label=label)
    ax.set_xlabel("Epsilon")
    ax.set_ylabel("IoU (Top 20%)")
    ax.set_title("Feature Overlap (IoU) vs Perturbation Size")
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, output_filename)
    plt.savefig(save_path)
    plt.close(fig)
    return save_path

def plot_accuracy_vs_eps(results, output_filename="accuracy_vs_eps.png"):
    """
    Plot Accuracy vs Epsilon for different attacks.
    results: list of result dictionaries
    """
    # Group results by Attack
    # We assume 'accuracy' key exists in results
    groups = {}
    for res in results:
        # Check if accuracy exists
        if 'accuracy' not in res:
            continue
            
        key = res['attack']
        if key not in groups:
            groups[key] = {'eps': [], 'accuracy': []}
        
        # Avoid duplicate points for same eps (since we have multiple methods per attack/eps)
        # We can just check if eps is already recorded? 
        # Actually, main.py loop structure: attack -> method -> eps.
        # So for a given attack and eps, we calculate accuracy MULTIPLE times (once per method loop).
        # We should only plot distinct (eps, acc) pairs per attack.
        eps_val = res['eps']
        acc_val = res['accuracy']
        
        # Check if we already have this eps for this attack
        if eps_val in groups[key]['eps']:
            continue
            
        groups[key]['eps'].append(eps_val)
        groups[key]['accuracy'].append(acc_val)
    
    # Sort
    for key in groups:
        sorted_indices = np.argsort(groups[key]['eps'])
        groups[key]['eps'] = np.array(groups[key]['eps'])[sorted_indices]
        groups[key]['accuracy'] = np.array(groups[key]['accuracy'])[sorted_indices]
        
    fig, ax = plt.subplots(figsize=(8, 6))
    for attack, data in groups.items():
        if len(data['eps']) > 0:
            ax.plot(data['eps'], data['accuracy'], marker='o', label=f"{attack.upper()}")
            
    ax.set_xlabel("Epsilon")
    ax.set_ylabel("Model Accuracy")
    ax.set_title("Robustness: Accuracy vs Perturbation Size")
    ax.legend()
    ax.grid(True)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, output_filename)
    plt.savefig(save_path)
    plt.close(fig)
    return save_path
