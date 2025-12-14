import torch
import numpy as np
import json
import os
from .config import DEVICE, RESULTS_DIR
from .model import get_resnet18_cifar10, get_dataloaders
from .attacks import fgsm_attack, pgd_attack
from .interpretability import compute_gradcam, compute_ig, compute_shap, get_explanation_maps
from .metrics import pair_metrics
from .utils import set_seed
from .visualization import save_saliency_comparison, plot_metrics_vs_eps, plot_accuracy_vs_eps

# ... (keep existing imports)

def explanation_metrics_for_eps(model,
                                loader,
                                attack="pgd",
                                method="gradcam",
                                eps_list=[0, 2/255, 4/255, 8/255],
                                max_images=200):
    """
    Returns list of dicts:
    { 'eps':..., 'attack':..., 'method':..., 'ssim':..., 'iou':..., 'accuracy':... }
    aggregated over images.
    """
    model.eval()
    results = []
    
    for eps in eps_list:
        metric_list = []
        imgs_processed = 0
        total_correct = 0
        total_samples = 0
        
        # Keep track of first image ID for saving example plots
        plot_saved = False
        
        for batch_idx, (x, y) in enumerate(loader):
            if imgs_processed >= max_images:
                break

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            # clean maps
            maps_clean = get_explanation_maps(model, x, method)

            # adversarial sample
            if attack.lower() == "fgsm":
                x_adv = fgsm_attack(model, x, y, eps)
            else:
                x_adv = pgd_attack(model, x, y, eps, alpha=2/255, steps=10)
            
            # Compute accuracy on adversarial examples (or clean if eps=0)
            with torch.no_grad():
                logits = model(x_adv)
                preds = logits.argmax(dim=1)
                total_correct += (preds == y).sum().item()
                total_samples += y.size(0)

            maps_adv = get_explanation_maps(model, x_adv, method)

            # per-image metrics
            for i in range(x.size(0)):
                m_clean = maps_clean[i]
                m_adv = maps_adv[i]
                metric_list.append(pair_metrics(m_clean, m_adv))
                
                # Save just one example per condition for report
                if not plot_saved and eps > 0 and i == 0: 
                    save_saliency_comparison(x[i], x_adv[i], m_clean, m_adv, method, attack, eps, f"batch{batch_idx}_img{i}")
                    plot_saved = True

            imgs_processed += x.size(0)

        # aggregate over all images for this eps
        if metric_list:
            keys = metric_list[0].keys()
            agg = {k: float(np.mean([m[k] for m in metric_list])) for k in keys}
            
            # Accuracy for this eps
            acc = total_correct / total_samples if total_samples > 0 else 0.0
            
            agg.update({
                "eps": float(eps),
                "attack": attack,
                "method": method,
                "n_images": len(metric_list),
                "accuracy": float(acc)
            })
            results.append(agg)
            print(f"[{method.upper()} | {attack.upper()}] eps={eps:.4f} -> Acc={acc*100:.2f}%, SSIM={agg['ssim']:.3f}")
    
    return results

def main():
    set_seed(42)
    
    # Initialize model
    print("Initializing model...")
    checkpoint = "models/resnet18_cifar10.pth"
    if not os.path.exists(checkpoint):
        print(f"Warning: Checkpoint {checkpoint} not found. Using untrained model.")
        checkpoint = None
    else:
        print(f"Loading trained weights from {checkpoint}")
        
    model = get_resnet18_cifar10(checkpoint_path=checkpoint)
    
    # Get dataloaders
    print("Loading data...")
    train_loader, test_loader = get_dataloaders(batch_size=32)
    
    # Define experiments
    eps_list = [0, 2/255, 4/255, 8/255, 12/255]
    attacks = ["fgsm", "pgd"]
    methods = ["gradcam", "ig", "shap"]
    
    all_results = []
    
    for attack in attacks:
        for method in methods:
            print(f"\nRunning experiment: Attack={attack}, Method={method}")
            res = explanation_metrics_for_eps(
                model,
                test_loader,
                attack=attack,
                method=method,
                eps_list=eps_list,
                max_images=20 # Small/Medium batch for visualization demo
            )
            all_results.extend(res)
            
    # Save results
    save_path = os.path.join(RESULTS_DIR, "experiment_results.json")
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {save_path}")
    
    # Generate final plots
    print("Generating summary plots...")
    plot_path1 = plot_metrics_vs_eps(all_results)
    print(f"Metrics plot saved to {plot_path1}")
    plot_path2 = plot_accuracy_vs_eps(all_results)
    print(f"Accuracy plot saved to {plot_path2}")

if __name__ == "__main__":
    main()
