import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.stats import entropy as entropy_fn, pearsonr

def map_to_heatmap(arr: np.ndarray) -> np.ndarray:
    """
    Normalize a saliency map to [0, 1]
    """
    arr = arr.astype(np.float32)
    arr = arr - arr.min()
    maxv = arr.max()
    if maxv > 0:
        arr = arr / maxv
    return arr

def compute_entropy_map(arr: np.ndarray) -> float:
    """
    Entropy of the normalized saliency map (flattened).
    """
    arr = arr.astype(np.float64)
    arr = arr - arr.min()
    s = arr.sum()
    if s <= 0:
        return 0.0
    p = arr / s
    p = p[p > 0]
    return float(entropy_fn(p, base=2))

def pair_metrics(a: np.ndarray, b: np.ndarray, top_prop: float = 0.2):
    """
    Compare two saliency maps (already normalized to [0,1]).
    Returns SSIM, IoU of top-k, entropy_clean, entropy_adv, correlation.
    """
    a = map_to_heatmap(a)
    b = map_to_heatmap(b)

    # SSIM (ensure 2D)
    s = ssim(a, b, data_range=1.0)

    # Top-proportion IoU
    thresh_a = np.percentile(a.flatten(), 100 * (1 - top_prop))
    thresh_b = np.percentile(b.flatten(), 100 * (1 - top_prop))
    mask_a = (a >= thresh_a)
    mask_b = (b >= thresh_b)
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union > 0:
        iou = inter / union
    else:
        iou = 1.0 if inter == 0 else 0.0

    # Entropies
    ent_a = compute_entropy_map(a)
    ent_b = compute_entropy_map(b)

    # Pearson correlation
    try:
        corr = pearsonr(a.flatten(), b.flatten())[0]
    except Exception:
        corr = 0.0

    return {
        "ssim": float(s),
        "iou": float(iou),
        "entropy_clean": float(ent_a),
        "entropy_adv": float(ent_b),
        "corr": float(corr),
    }
