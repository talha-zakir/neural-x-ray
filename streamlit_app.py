import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# Import project modules
from src.config import DEVICE, CIFAR_MEAN, CIFAR_STD
from src.model import get_resnet18_cifar10, get_dataloaders
from src.attacks import fgsm_attack, pgd_attack
from src.interpretability import get_explanation_maps
from src.metrics import pair_metrics
from src.utils import denormalize_cifar

# Page Config
st.set_page_config(
    page_title="Neural X-Ray",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Dark Theme CSS
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid #0f3460;
    }
    
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label {
        color: #e0e0e0 !important;
        font-weight: 500;
    }
    
    /* Header Styles */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.85);
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
    }
    
    /* Card Styles */
    .metric-card {
        background: linear-gradient(145deg, #1e1e2f 0%, #2d2d44 100%);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.4);
    }
    
    .metric-card h3 {
        color: #a0a0a0;
        font-size: 0.85rem;
        font-weight: 500;
        margin: 0 0 0.5rem 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-card .value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .metric-card .delta {
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    .value-green { color: #00d26a; }
    .value-red { color: #ff4757; }
    .value-blue { color: #667eea; }
    .value-orange { color: #ffa502; }
    
    .delta-positive { color: #00d26a; }
    .delta-negative { color: #ff4757; }
    
    /* Section Headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid rgba(102, 126, 234, 0.3);
    }
    
    .section-header h2 {
        color: #e0e0e0;
        font-size: 1.25rem;
        font-weight: 600;
        margin: 0;
    }
    
    .section-icon {
        font-size: 1.5rem;
    }
    
    /* Image Container */
    .image-container {
        background: #1e1e2f;
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid rgba(255,255,255,0.05);
    }
    
    .image-label {
        color: #a0a0a0;
        font-size: 0.85rem;
        text-align: center;
        margin-top: 0.5rem;
    }
    
    /* Status Badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .status-standard {
        background: rgba(102, 126, 234, 0.2);
        color: #667eea;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    .status-robust {
        background: rgba(0, 210, 106, 0.2);
        color: #00d26a;
        border: 1px solid rgba(0, 210, 106, 0.3);
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Divider */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(102,126,234,0.5), transparent);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

CIFAR_CLASSES = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

@st.cache_resource
def load_model(model_type: str = "Standard"):
    """Load the model with specified type."""
    if model_type == "Robust (Adv. Trained)":
        checkpoint = "models/resnet18_cifar10_adv.pth"
    else:
        checkpoint = "models/resnet18_cifar10.pth"
    
    try:
        model = get_resnet18_cifar10(checkpoint_path=checkpoint)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        model = get_resnet18_cifar10(checkpoint_path=None)
    
    model.eval()
    return model

@st.cache_resource
def load_data():
    """Load a batch of test images once."""
    _, test_loader = get_dataloaders(batch_size=64)
    return next(iter(test_loader))

def create_heatmap_figure(img, heatmap):
    """Create a clean heatmap overlay figure."""
    fig, ax = plt.subplots(figsize=(4, 4), facecolor='#1e1e2f')
    ax.imshow(img)
    ax.imshow(heatmap, cmap='magma', alpha=0.6)
    ax.axis('off')
    plt.tight_layout(pad=0)
    return fig

def metric_card(label, value, delta=None, color="blue"):
    """Generate HTML for a metric card."""
    delta_html = ""
    if delta is not None:
        delta_class = "delta-positive" if delta >= 0 else "delta-negative"
        delta_sign = "+" if delta >= 0 else ""
        delta_html = f'<p class="delta {delta_class}">{delta_sign}{delta:.1f}%</p>'
    
    return f"""
    <div class="metric-card">
        <h3>{label}</h3>
        <p class="value value-{color}">{value}</p>
        {delta_html}
    </div>
    """

def main():
    # --- Header ---
    st.markdown("""
    <div class="main-header">
        <h1>🔬 Neural X-Ray</h1>
        <p>Adversarial Forensics Lab — Visualize how neural networks respond to attacks</p>
    </div>
    """, unsafe_allow_html=True)
    
    # --- Sidebar ---
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.markdown("---")
        
        model_type = st.selectbox(
            "🧠 Model",
            ["Standard", "Robust (Adv. Trained)"],
            help="Standard: Regular training. Robust: Trained with adversarial examples."
        )
        
        img_index = st.slider("🖼️ Image ID", 0, 31, 0)
        
        attack_method = st.selectbox("⚔️ Attack", ["FGSM", "PGD"])
        
        eps = st.slider("💥 Epsilon", 0.0, 0.10, 0.0, 0.001, 
                        help="Perturbation strength (0 = no attack)")
        
        interp_method = st.selectbox(
            "🔍 Explanation",
            ["GradCAM", "Integrated Gradients", "SHAP"],
            help="Method for visualizing model attention"
        )
        
        st.markdown("---")
        
        # Model Status Badge
        if model_type == "Robust (Adv. Trained)":
            st.markdown('<div class="status-badge status-robust">🛡️ Robust Model</div>', 
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-badge status-standard">📊 Standard Model</div>', 
                        unsafe_allow_html=True)
    
    # Map method names
    method_map = {"GradCAM": "gradcam", "Integrated Gradients": "ig", "SHAP": "shap"}
    method_code = method_map[interp_method]
    
    # --- Load Resources ---
    with st.spinner("Loading model..."):
        model = load_model(model_type)
        images, labels = load_data()
    
    # Get image
    img_tensor = images[img_index].unsqueeze(0).to(DEVICE)
    label_idx = labels[img_index].item()
    true_label = CIFAR_CLASSES[label_idx]
    
    # --- Predictions ---
    with torch.no_grad():
        clean_logits = model(img_tensor)
        clean_probs = torch.softmax(clean_logits, dim=1)[0]
        clean_conf = clean_probs[label_idx].item()
        clean_pred_idx = clean_logits.argmax().item()
    
    clean_map = get_explanation_maps(model, img_tensor, method_code)[0]
    
    # Attack
    if eps > 0:
        if attack_method == "FGSM":
            adv_tensor = fgsm_attack(model, img_tensor, labels[img_index].unsqueeze(0).to(DEVICE), eps)
        else:
            # alpha = step size, typically eps/steps or eps/4
            adv_tensor = pgd_attack(model, img_tensor, labels[img_index].unsqueeze(0).to(DEVICE), 
                                   eps=eps, alpha=eps/4, steps=10)
    else:
        adv_tensor = img_tensor
    
    with torch.no_grad():
        adv_logits = model(adv_tensor)
        adv_probs = torch.softmax(adv_logits, dim=1)[0]
        adv_conf = adv_probs[label_idx].item()
        adv_pred_idx = adv_logits.argmax().item()
        adv_pred_label = CIFAR_CLASSES[adv_pred_idx]
    
    adv_map = get_explanation_maps(model, adv_tensor, method_code)[0]
    
    # Metrics
    metrics = pair_metrics(clean_map, adv_map)
    ssim_val = metrics['ssim']
    
    # Display helpers
    def to_display(t):
        return denormalize_cifar(t).cpu().numpy()[0].transpose(1, 2, 0)
    
    img_clean = to_display(img_tensor)
    img_adv = to_display(adv_tensor)
    noise = np.abs(img_adv - img_clean)
    noise = noise / noise.max() if noise.max() > 0 else noise
    
    # --- Key Metrics Row ---
    st.markdown('<div class="section-header"><span class="section-icon">📊</span><h2>Key Metrics</h2></div>', 
                unsafe_allow_html=True)
    
    m1, m2, m3, m4 = st.columns(4)
    
    with m1:
        st.markdown(metric_card("True Class", true_label.upper(), color="blue"), unsafe_allow_html=True)
    
    with m2:
        color = "green" if clean_conf > 0.5 else "orange"
        st.markdown(metric_card("Clean Confidence", f"{clean_conf*100:.1f}%", color=color), unsafe_allow_html=True)
    
    with m3:
        delta = (adv_conf - clean_conf) * 100
        color = "green" if adv_conf > 0.5 else "red"
        st.markdown(metric_card("After Attack", f"{adv_conf*100:.1f}%", delta=delta, color=color), unsafe_allow_html=True)
    
    with m4:
        color = "green" if ssim_val > 0.7 else ("orange" if ssim_val > 0.4 else "red")
        st.markdown(metric_card("Stability (SSIM)", f"{ssim_val:.2f}", color=color), unsafe_allow_html=True)
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # --- Visualization Section ---
    st.markdown('<div class="section-header"><span class="section-icon">🔬</span><h2>Visual Analysis</h2></div>', 
                unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("##### 🛡️ Original Image")
        st.image(img_clean, use_container_width=True, caption=f"Class: {true_label}")
        
        st.markdown("##### Attention Map")
        fig = create_heatmap_figure(img_clean, clean_map)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    
    with col2:
        attack_label = f"{attack_method} (ε={eps})" if eps > 0 else "No Attack"
        st.markdown(f"##### ⚔️ {attack_label}")
        st.image(img_adv, use_container_width=True, caption=f"Prediction: {adv_pred_label}")
        
        st.markdown("##### Attention Map")
        fig = create_heatmap_figure(img_adv, adv_map)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    
    with col3:
        st.markdown("##### 🔊 Perturbation")
        st.image(noise, use_container_width=True, caption="Amplified Noise", clamp=True)
        
        st.markdown("##### Attack Info")
        st.markdown(f"""
        <div class="metric-card" style="text-align: left; padding: 1rem;">
            <p style="color: #a0a0a0; margin: 0.5rem 0;"><strong>Method:</strong> {attack_method}</p>
            <p style="color: #a0a0a0; margin: 0.5rem 0;"><strong>Epsilon:</strong> {eps}</p>
            <p style="color: #a0a0a0; margin: 0.5rem 0;"><strong>Original:</strong> {true_label}</p>
            <p style="color: #a0a0a0; margin: 0.5rem 0;"><strong>Predicted:</strong> {adv_pred_label}</p>
            <p style="color: {'#00d26a' if adv_pred_label == true_label else '#ff4757'}; margin: 0.5rem 0;">
                <strong>{'✓ Correct' if adv_pred_label == true_label else '✗ Fooled'}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
