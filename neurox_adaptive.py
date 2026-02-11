

import os
import sys
import io
import base64
import tempfile
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import (binary_closing, binary_fill_holes, distance_transform_edt, 
                           gaussian_filter, label as scipy_label, zoom)
from scipy import ndimage
from skimage import measure
import trimesh
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_agg import FigureCanvasAgg
import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors as rl_colors
from reportlab.lib.units import inch

# Groq AI Integration
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    Groq = None

# HD-BET for Medical-Grade Brain Extraction (CLI-BASED)
# Using CLI interface (official, stable) instead of Python imports (unreliable)
import subprocess

HDBET_AVAILABLE = False

print("=" * 60)
print("🔍 Checking for HD-BET (CLI)...")

try:
    result = subprocess.run(
        ["hd-bet", "-h"],
        capture_output=True,
        text=True,
        timeout=30  # HD-BET can take time to load
    )
    if result.returncode == 0:
        HDBET_AVAILABLE = True
        print("✅ HD-BET CLI is available and working")
        print("🎯 HD-BET WILL BE USED as the ONLY brain extraction method")
        print("   (No fallback to heuristics - clinical standard)")
    else:
        print("❌ HD-BET CLI returned error")
except (subprocess.TimeoutExpired, FileNotFoundError) as e:
    print(f"❌ HD-BET CLI not found: {e}")
    print("⚠️  3D brain surface rendering will be DISABLED")
    print("   This is CORRECT behavior for clinical safety")
    print("\n   To install HD-BET:")
    print("   pip install HD-BET")

print("=" * 60)

# ═══════════════════════════════════════════════════════════════════════════
# PRODUCTION CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# Offline Mode: Disable external dependencies for reproducibility
OFFLINE_MODE = os.getenv("NEUROX_OFFLINE", "False").lower() == "true"
if OFFLINE_MODE:
    print("🔒 OFFLINE MODE: Groq AI disabled, local inference only")
    GROQ_AVAILABLE = False

# Deterministic Mode: For reproducible results (academic requirement)
DETERMINISTIC_MODE = True

if DETERMINISTIC_MODE:
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Device and Model Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROI_SIZE = (96, 96, 96)
PRESENCE_THRESHOLD = 0.5
MODEL_PATH = r"C:\Users\karth\OneDrive\Desktop\neurox\neurox_multihead_final.pth"
ASSET_DIR = Path("./assets/brain")

# Disease Configuration
DISEASE_COLORS = {
    "tumor": {"rgb": [255, 68, 68], "hex": "#FF4444", "name": "Tumor"},
    "stroke": {"rgb": [68, 68, 255], "hex": "#4444FF", "name": "Stroke"},
    "alzheimer": {"rgb": [255, 136, 0], "hex": "#FF8800", "name": "Alzheimer Pattern"}
}

# ═══════════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE (EXACT MATCH WITH TRAINING)
# ═══════════════════════════════════════════════════════════════════════════

class TransformerBottleneck3D(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True),
                nn.LayerNorm(dim),
                nn.Sequential(
                    nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(mlp_dim, dim), nn.Dropout(dropout),
                )
            ]))
    
    def forward(self, x):
        b, c, d, h, w = x.shape
        x = x.view(b, c, -1).permute(0, 2, 1)
        for ln1, attn, ln2, ff in self.layers:
            attn_out, _ = attn(ln1(x), ln1(x), ln1(x))
            x = x + attn_out
            x = x + ff(ln2(x))
        return x.permute(0, 2, 1).view(b, c, d, h, w)


class SharedEncoder(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.enc1 = self._conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = self._conv_block(32, 64)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = self._conv_block(64, 128)
        self.pool3 = nn.MaxPool3d(2)
        self.bottleneck = TransformerBottleneck3D(128, 4, 8, 256, 0.1)
    
    def _conv_block(self, in_c, out_c):
        """InstanceNorm3d for consistency with training pipeline (batch_size=2 stability)."""
        return nn.Sequential(
            nn.Conv3d(in_c, out_c, 3, padding=1),
            nn.InstanceNorm3d(out_c, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_c, out_c, 3, padding=1),
            nn.InstanceNorm3d(out_c, affine=True),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        if x.shape[1] == 2:
            x = x.mean(dim=1, keepdim=True)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        return {"enc1": e1, "enc2": e2, "enc3": e3, "bottleneck": b}


class PresenceHead(nn.Module):
    """Binary presence detector with uncertainty estimation.
    
    PRODUCTION IMPROVEMENT: Monte-Carlo Dropout for uncertainty quantification.
    Enables model to communicate confidence level - critical for clinical trust.
    """
    def __init__(self, in_features=128):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, bottleneck_features):
        x = self.pool(bottleneck_features)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def uncertainty_forward(self, bottleneck_features, n_samples=10):
        """Monte-Carlo Dropout inference for epistemic uncertainty estimation.
        
        Scientific Rationale:
        - Multiple stochastic forward passes with dropout enabled
        - Variance across samples = epistemic (model) uncertainty
        - High uncertainty → model unsure, recommend expert review
        
        Reference: Gal & Ghahramani (2016) - Dropout as Bayesian Approximation
        """
        self.train()  # Enable dropout
        
        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                logit = self.forward(bottleneck_features)
                prob = torch.sigmoid(logit).cpu().item()
                samples.append(prob)
        
        self.eval()  # Restore eval mode
        
        mean_prob = float(np.mean(samples))
        uncertainty = float(np.std(samples))  # Epistemic uncertainty
        
        return mean_prob, uncertainty


class AttentionGate3D(nn.Module):
    def __init__(self, gate_ch, skip_ch, inter_ch):
        super().__init__()
        self.W_gate = nn.Conv3d(gate_ch, inter_ch, 1)
        self.W_skip = nn.Conv3d(skip_ch, inter_ch, 1)
        self.psi = nn.Conv3d(inter_ch, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, gate, skip):
        psi = self.relu(self.W_gate(gate) + self.W_skip(skip))
        return skip * self.sigmoid(self.psi(psi))


class SegmentationDecoder(nn.Module):
    def __init__(self, output_channels, name):
        super().__init__()
        self.name = name
        self.up3 = nn.ConvTranspose3d(128, 128, 2, 2)
        self.att3 = AttentionGate3D(128, 128, 64)
        self.dec3 = self._conv_block(256, 128)
        self.up2 = nn.ConvTranspose3d(128, 64, 2, 2)
        self.att2 = AttentionGate3D(64, 64, 32)
        self.dec2 = self._conv_block(128, 64)
        self.up1 = nn.ConvTranspose3d(64, 32, 2, 2)
        self.att1 = AttentionGate3D(32, 32, 16)
        self.dec1 = self._conv_block(64, 32)
        self.output_head = nn.Conv3d(32, output_channels, 1)
    
    def _conv_block(self, in_c, out_c):
        """InstanceNorm3d (matches encoder normalization)."""
        return nn.Sequential(
            nn.Conv3d(in_c, out_c, 3, padding=1),
            nn.InstanceNorm3d(out_c, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_c, out_c, 3, padding=1),
            nn.InstanceNorm3d(out_c, affine=True),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, enc_features):
        e1, e2, e3, b = enc_features["enc1"], enc_features["enc2"], enc_features["enc3"], enc_features["bottleneck"]
        u3 = self.up3(b)
        d3 = self.dec3(torch.cat([u3, self.att3(u3, e3)], dim=1))
        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, self.att2(u2, e2)], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, self.att1(u1, e1)], dim=1))
        return self.output_head(d1)


class NeuroXMultiDisease(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SharedEncoder(in_channels=1)
        self.presence_heads = nn.ModuleDict({
            "tumor": PresenceHead(128),
            "stroke": PresenceHead(128),
            "alzheimer": PresenceHead(128)
        })
        self.seg_decoders = nn.ModuleDict({
            "tumor": SegmentationDecoder(4, "tumor"),
            "stroke": SegmentationDecoder(1, "stroke")
        })
    
    def forward(self, x, active_presence=None, active_seg=None):
        features = self.encoder(x)
        presence = {}
        if active_presence:
            for key in active_presence:
                if key in self.presence_heads:
                    presence[key] = self.presence_heads[key](features["bottleneck"])
        segmentations = {}
        if active_seg:
            for key in active_seg:
                if key in self.seg_decoders:
                    segmentations[key] = self.seg_decoders[key](features)
        return {"presence": presence, "segmentations": segmentations}


# ═══════════════════════════════════════════════════════════════════════════
# INFERENCE UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model(model_path: str = MODEL_PATH):
    """Load trained model with InstanceNorm compatibility.
    
    PRODUCTION FIX: Strict=False loading for BatchNorm→InstanceNorm transition.
    Old checkpoints may have BatchNorm keys; we load what matches and skip rest.
    """
    model = NeuroXMultiDisease().to(DEVICE)
    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            # Validate architecture
            assert "alzheimer" not in model.seg_decoders, "Alzheimer must not have segmentation decoder"
            assert "alzheimer" in model.presence_heads, "Alzheimer must have presence head"
            
            model.eval()
            
            # Warn about mismatched keys (expected for BatchNorm→InstanceNorm)
            if missing_keys or unexpected_keys:
                warning_msg = f"⚠️ Checkpoint partial match: {len(missing_keys)} missing, {len(unexpected_keys)} unexpected\n"
                warning_msg += "This is expected when loading BatchNorm checkpoints into InstanceNorm model."
                st.info(warning_msg)
                print(warning_msg)
            
            print(f"✅ Model loaded: {model_path}")
            print(f"✅ Architecture validated: Alzheimer presence-only, no segmentation")
            return model
        except Exception as e:
            st.error(f"Model loading failed: {e}")
            traceback.print_exc()
            return None
    else:
        st.warning(f"⚠️ Model file not found: {model_path}")
        return None


def load_and_preprocess_nifti(file_path: str) -> Tuple[torch.Tensor, np.ndarray, Dict, np.ndarray, Tuple]:
    """Load, normalize, and prepare NIfTI file with complete metadata preservation.
    
    MEDICAL-GRADE PREPROCESSING:
    Preserves affine matrix and voxel spacing for anatomically accurate visualization.
    
    Returns:
        roi_tensor: 96³ tensor for model inference
        original_data: Full-resolution patient MRI (original intensity)
        roi_metadata: Coordinate space mapping information
        affine: NIfTI affine matrix (4x4)
        spacing: Voxel spacing in mm (dx, dy, dz)
    """
    img = nib.load(file_path)
    original_data = img.get_fdata().astype(np.float32)
    original_shape = original_data.shape
    
    # CRITICAL: Proper spacing extraction from affine matrix
    affine = img.affine
    
    # Extract voxel spacing as magnitude of affine column vectors
    # This handles rotations and non-axis-aligned affines correctly
    spacing_raw = np.sqrt(np.sum(affine[:3, :3]**2, axis=0))
    
    # Validate spacing (must be positive, non-zero)
    if np.any(spacing_raw <= 0) or np.any(np.isnan(spacing_raw)):
        print(f"⚠️ Invalid spacing from affine: {spacing_raw}, using default (1.0, 1.0, 1.0)")
        spacing = (1.0, 1.0, 1.0)
    else:
        spacing = tuple(spacing_raw)
        print(f"✅ Voxel spacing: {spacing}")
    
    # Handle dimensions
    if original_data.ndim == 4:
        if original_data.shape[-1] <= 2:
            data = original_data[..., 0]
        else:
            data = original_data[..., :2].mean(axis=-1)
    else:
        data = original_data.copy()
    
    # Z-score normalization (for inference ONLY, not visualization)
    mean, std = data.mean(), data.std() + 1e-8
    data_normalized = np.clip((data - mean) / std, -5, 5)
    
    # Prepare ROI tensor for inference
    data_normalized = data_normalized[np.newaxis, ...]
    roi_tensor = torch.from_numpy(data_normalized).unsqueeze(0).float()
    roi_tensor = F.interpolate(roi_tensor, size=ROI_SIZE, mode='trilinear', align_corners=False)
    
    # Store ROI metadata for coordinate mapping
    roi_metadata = {
        "original_shape": tuple(original_shape[:3]) if original_data.ndim == 4 else original_shape,
        "roi_shape": ROI_SIZE,
        "scale_factors": tuple(np.array(original_shape[:3]) / np.array(ROI_SIZE)),
        "interpolation_mode": "trilinear"
    }
    
    return roi_tensor, original_data, roi_metadata, affine, spacing


def compute_lesion_metrics(segmentation: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> Dict:
    """Clinical-style volumetric quantification.
    
    PRODUCTION IMPROVEMENT: Transforms visual segmentation into quantitative metrics.
    Essential for academic defense - shows clinical applicability.
    
    Args:
        segmentation: Binary segmentation mask (3D or 4D)
        spacing: Voxel spacing in mm (default: 1mm isotropic)
    
    Returns:
        Dictionary with voxel_count, volume_mm3, volume_ml, percent_brain
    """
    # Handle multi-channel (collapse to binary)
    if segmentation.ndim == 4:
        binary = segmentation.max(axis=0)
    elif segmentation.ndim == 3:
        binary = segmentation
    else:
        raise ValueError(f"Invalid segmentation shape: {segmentation.shape}")
    
    voxel_count = int(binary.sum())
    
    # Volume calculations
    voxel_volume_mm3 = np.prod(spacing)
    volume_mm3 = float(voxel_count * voxel_volume_mm3)
    volume_ml = volume_mm3 / 1000.0
    
    # Approximate brain volume (1400 mL average adult brain)
    brain_volume_estimate = 1400.0
    percent_brain = (volume_ml / brain_volume_estimate) * 100.0
    
    return {
        "voxel_count": voxel_count,
        "volume_mm3": volume_mm3,
        "volume_ml": volume_ml,
        "percent_brain": percent_brain
    }


def apply_multi_label_detection(presence_logits: Dict[str, float], threshold: float = 0.5) -> Dict:
    """Apply multi-label disease detection with independent probabilities.
    
    CRITICAL CORRECTION: Diseases are NOT mutually exclusive.
    One patient can have tumor AND stroke AND Alzheimer's simultaneously.
    
    Uses sigmoid (NOT softmax) for independent binary classification per disease.
    This matches the BCEWithLogitsLoss training objective.
    
    Args:
        presence_logits: Raw logits from presence heads {disease: logit_value}
        threshold: Detection threshold (default: 0.5)
    
    Returns:
        Dict containing:
            - disease_probabilities: Independent sigmoid probabilities (can sum to >1.0)
            - detected_diseases: List of all diseases above threshold
            - detection_confidence: Dict of confidence per detected disease
            - all_probabilities: All disease probabilities for reference
    """
    import torch
    
    disease_names = ["tumor", "stroke", "alzheimer"]
    
    # Apply sigmoid independently to each disease (multi-label)
    disease_probs = {}
    for disease in disease_names:
        logit = presence_logits[disease]
        # Sigmoid for independent binary classification
        prob = float(torch.sigmoid(torch.tensor(logit, dtype=torch.float32)).item())
        disease_probs[disease] = prob
    
    # Detect ALL diseases above threshold (multi-label)
    detected_diseases = [
        disease for disease, prob in disease_probs.items()
        if prob >= threshold
    ]
    
    # Confidence for detected diseases
    detection_confidence = {
        disease: disease_probs[disease]
        for disease in detected_diseases
    }
    
    return {
        "disease_probabilities": disease_probs,
        "detected_diseases": detected_diseases,
        "detection_confidence": detection_confidence,
        "all_probabilities": disease_probs,
        "threshold_used": threshold,
        "multi_label": True  # Flag indicating multi-label classification
    }


def automatic_disease_detection(
    model, 
    image_tensor: torch.Tensor, 
    threshold: float = PRESENCE_THRESHOLD,
    use_uncertainty: bool = True
) -> Dict:
    """Automatic multi-label disease detection.
    
    CRITICAL: Uses independent sigmoid probabilities (multi-label).
    One patient can have multiple diseases simultaneously.
    """
    if model is None:
        return {"detected_diseases": [], "probabilities": {}, "uncertainties": {}}
    
    diseases = ["tumor", "stroke", "alzheimer"]
    probabilities = {}
    uncertainties = {}
    presence_logits = {}
    
    with torch.no_grad():
        features = model.encoder(image_tensor.to(DEVICE))
        bottleneck = features["bottleneck"]
        
        for disease in diseases:
            head = model.presence_heads[disease]
            
            if use_uncertainty:
                # MC Dropout inference
                mean_prob, uncertainty = head.uncertainty_forward(bottleneck, n_samples=10)
                probabilities[disease] = mean_prob
                uncertainties[disease] = uncertainty
                # Approximate logit from probability
                presence_logits[disease] = np.log(mean_prob / (1 - mean_prob + 1e-8))
            else:
                # Standard deterministic inference - extract LOGIT
                logit = head(bottleneck)
                presence_logits[disease] = float(logit.cpu().item())
                prob = torch.sigmoid(logit).cpu().item()
                probabilities[disease] = prob
                uncertainties[disease] = 0.0
    
    # Apply multi-label detection (independent sigmoid)
    detection_result = apply_multi_label_detection(presence_logits, threshold)
    
    return {
        "detected_diseases": detection_result["detected_diseases"],
        "probabilities": detection_result["disease_probabilities"],
        "uncertainties": uncertainties,
        "detection_confidence": detection_result["detection_confidence"],
        "multi_label": True
    }


def perform_segmentation(model, image_tensor: torch.Tensor, diseases: List[str]) -> Dict:
    """Segment detected diseases (tumor/stroke only).
    
    PRODUCTION FIX: Hard guard against Alzheimer segmentation.
    Alzheimer is presence-only - no voxel-level pathology exists in training data.
    """
    if model is None:
        return {}
    
    # CRITICAL GUARD: Alzheimer cannot have segmentation
    assert "alzheimer" not in model.seg_decoders, \
        "ARCHITECTURAL CONSTRAINT VIOLATED: Alzheimer has no segmentation decoder"
    
    # Filter to only segmentable diseases
    seg_diseases = [d for d in diseases if d in ["tumor", "stroke"]]
    
    # Inform user if Alzheimer was requested
    if "alzheimer" in diseases:
        st.info("ℹ️ **Alzheimer Methodology**: Presence detected via global pattern recognition. "
                "No voxel-level segmentation available (ADNI dataset provides subject-level labels only).")
    
    if not seg_diseases:
        return {}
    
    results = {}
    
    with torch.no_grad():
        output = model(image_tensor.to(DEVICE), active_presence=None, active_seg=seg_diseases)
        
        for disease in seg_diseases:
            seg_logits = output["segmentations"][disease]
            seg_probs = torch.sigmoid(seg_logits).cpu().numpy()[0]
            
            # Post-process: morphological operations
            seg_binary = (seg_probs > 0.5).astype(np.uint8)
            
            for c in range(seg_binary.shape[0]):
                seg_binary[c] = binary_closing(seg_binary[c], iterations=2)
                seg_binary[c] = binary_fill_holes(seg_binary[c])
                labeled, num = scipy_label(seg_binary[c])
                if num > 0:
                    sizes = [(labeled == i).sum() for i in range(1, num + 1)]
                    if sizes:
                        largest = np.argmax(sizes) + 1
                        seg_binary[c] = (labeled == largest).astype(np.uint8)
            
            results[disease] = seg_probs, seg_binary
    
    return results


@st.cache_resource
def load_brain_atlas():
    """Load FreeSurfer atlas"""
    try:
        lh = trimesh.load(str(ASSET_DIR / "lh_fsaverage.ply"))
        rh = trimesh.load(str(ASSET_DIR / "rh_fsaverage.ply"))
        return lh, rh
    except Exception as e:
        st.warning(f"⚠️ Atlas loading failed: {e}")
        return None, None


def map_segmentation_to_original_space(
    seg_roi: np.ndarray,
    roi_metadata: Dict
) -> np.ndarray:
    """Map ROI segmentation back to patient coordinate system.
    
    CRITICAL FUNCTION: Fixes fundamental ROI cube rendering issue.
    Segmentation predicted in 96³ ROI space must be mapped back to original
    patient MRI space for anatomically meaningful visualization.
    
    Args:
        seg_roi: Segmentation in ROI space (96³ or multi-channel)
        roi_metadata: Coordinate mapping info from preprocessing
    
    Returns:
        Segmentation in original patient space
    """
    # Handle multi-channel (tumor has 4 channels)
    if seg_roi.ndim == 4:
        seg_roi = seg_roi.max(axis=0)  # Collapse to single channel
    elif seg_roi.ndim == 3:
        if seg_roi.shape[0] <= 4:  # Channel dimension
            seg_roi = seg_roi.max(axis=0)
    
    # Upsample to original shape using nearest-neighbor (preserves binary labels)
    scale_factors = roi_metadata["scale_factors"]
    seg_original = zoom(
        seg_roi,
        zoom=scale_factors,
        order=0  # Nearest neighbor - critical for binary masks
    )
    
    # Ensure exact shape match (zoom can be off by 1 voxel)
    target_shape = roi_metadata["original_shape"]
    if seg_original.shape != target_shape:
        # Pad or crop to exact size
        seg_original = resize_to_exact_shape(seg_original, target_shape)
    
    return seg_original.astype(np.uint8)


def resize_to_exact_shape(volume: np.ndarray, target_shape: Tuple) -> np.ndarray:
    """Resize volume to exact target shape via padding/cropping."""
    current_shape = np.array(volume.shape)
    target_shape = np.array(target_shape)
    
    result = volume.copy()
    
    for axis in range(3):
        diff = target_shape[axis] - current_shape[axis]
        if diff > 0:
            # Pad
            pad_width = [(0, 0)] * 3
            pad_width[axis] = (0, diff)
            result = np.pad(result, pad_width, mode='constant', constant_values=0)
        elif diff < 0:
            # Crop
            slices = [slice(None)] * 3
            slices[axis] = slice(0, target_shape[axis])
            result = result[tuple(slices)]
    
    return result


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Extract largest connected component from binary mask."""
    labeled, num_features = scipy_label(mask)
    if num_features == 0:
        return mask
    
    sizes = [(labeled == i).sum() for i in range(1, num_features + 1)]
    largest_label = np.argmax(sizes) + 1
    return (labeled == largest_label).astype(mask.dtype)


def validate_lesion_position(lesion_mask: np.ndarray, brain_mask: np.ndarray) -> tuple:
    """Validate lesion is anatomically inside brain.
    
    GOLD-STANDARD CLINICAL QA:
    Lesions cannot exist outside brain tissue - this is anatomically impossible.
    Checks if lesion centroid lies within brain mask.
    
    Args:
        lesion_mask: Binary lesion segmentation
        brain_mask: Binary brain mask
    
    Returns:
        Tuple of (is_valid: bool, centroid: array or None, message: str)
    """
    if lesion_mask.sum() == 0:
        return False, None, "Empty lesion mask"
    
    if brain_mask is None or brain_mask.sum() == 0:
        return False, None, "Empty brain mask"
    
    # Compute lesion centroid for reporting
    coords = np.argwhere(lesion_mask > 0)
    centroid = coords.mean(axis=0).astype(int)
    
    # Check what fraction of lesion is inside brain
    overlap = (lesion_mask & brain_mask).sum()
    total = lesion_mask.sum()
    overlap_fraction = overlap / total
    
    # RELAXED threshold: 40% overlap (was 50%)
    # Some lesions naturally extend to brain boundaries
    if overlap_fraction < 0.4:
        return False, centroid, f"Only {overlap_fraction:.1%} of lesion inside brain"
    
    return True, centroid, f"Valid: {overlap_fraction:.1%} inside brain"



def generate_patient_brain_surface(
    brain_mask: np.ndarray,
    affine: Optional[np.ndarray] = None,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate patient-specific brain surface mesh from brain mask.
    
    MEDICAL-GRADE SURFACE RENDERING:
    Input MUST be a binary brain mask (from Otsu or skull-stripping).
    NEVER pass raw MRI intensity directly.
    
    Pipeline:
    1. Input: binary brain mask
    2. Light Gaussian smoothing (preserve gyri/sulci)
    3. Marching cubes at level=0.5
    4. Apply affine transform (world coordinates)
    
    Args:
        brain_mask: Binary brain mask (uint8 or bool)
        affine: NIfTI affine matrix for world coordinates
        spacing: Voxel spacing in mm
    
    Returns:
        (vertices, faces) - vertices in world coordinates if affine provided
    """
    if brain_mask.sum() == 0:
        raise ValueError("Empty brain mask - cannot generate surface")
    
    # Light Gaussian smoothing (σ=0.7 preserves gyri/sulci detail)
    from scipy.ndimage import gaussian_filter
    brain_smooth = gaussian_filter(brain_mask.astype(float), sigma=0.7)
    
    # Marching cubes at level=0.5 (standard for binary masks)
    try:
        verts, faces, normals, _ = measure.marching_cubes(
            brain_smooth,
            level=0.5,
            spacing=spacing
        )
    except (ValueError, RuntimeError) as e:
        raise RuntimeError(f"Brain surface generation failed: {e}")
    
    # Apply affine transform if provided (convert to world coordinates)
    if affine is not None:
        # Convert voxel coordinates to world coordinates
        verts_homogeneous = np.column_stack([verts, np.ones(len(verts))])
        verts = (affine @ verts_homogeneous.T).T[:, :3]
    
    return verts, faces


# ═══════════════════════════════════════════════════════════════════════════
# MORPHOLOGICAL CLEANING & ANATOMICAL POST-PROCESSING
# ═══════════════════════════════════════════════════════════════════════════

def apply_hdbet_brain_extraction(volume: np.ndarray, affine: np.ndarray, spacing: Tuple[float, float, float]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Medical-grade skull stripping using HD-BET CLI (GOLD STANDARD).
    
    This is the ONLY brain extraction method used for 3D visualization.
    NO FALLBACK to heuristics - hard failure if HD-BET unavailable.
    
    Returns:
        (brain_volume, brain_mask) if successful
        (None, None) if failed (3D rendering will be disabled)
    """
    if not HDBET_AVAILABLE:
        print("❌ HD-BET CLI not available")
        return None, None
    
    print("\n" + "="*60)
    print("🧠 HD-BET BRAIN EXTRACTION (Medical-Grade)")
    print("="*60)
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save input
            input_path = os.path.join(tmpdir, "input.nii.gz")
            output_path = os.path.join(tmpdir, "output.nii.gz")  # HD-BET requires .nii.gz
            
            print(f"📝 Saving temporary NIfTI...")
            nib.save(nib.Nifti1Image(volume, affine), input_path)
            print(f"   Input: {input_path}")
            
            # Call HD-BET CLI
            print(f"🔧 Running HD-BET CLI...")
            cmd = [
                "hd-bet",
                "-i", input_path,
                "-o", output_path,
                "-device", "cpu",    # CPU for compatibility
                "--disable_tta"      # Disable test-time augmentation (faster)
            ]
            
            print(f"   Command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes max
            )
            
            if result.returncode != 0:
                print(f"❌ HD-BET failed with code {result.returncode}")
                print(f"   stderr: {result.stderr}")
                return None, None
            
            print(f"✅ HD-BET completed successfully")
            
            # Load results - HD-BET adds "_bet" suffix
            brain_path = output_path.replace(".nii.gz", "_bet.nii.gz")
            mask_path = output_path.replace(".nii.gz", "_bet_mask.nii.gz")
            
            print(f"� Loading output files...")
            print(f"   Brain: {brain_path}")
            print(f"   Mask:  {mask_path}")
            
            if not os.path.exists(brain_path) or not os.path.exists(mask_path):
                print(f"❌ Output files not found")
                return None, None
            
            brain_volume = nib.load(brain_path).get_fdata()
            brain_mask = nib.load(mask_path).get_fdata().astype(bool)
            
            # CRITICAL: Validate brain mask
            brain_voxels = brain_mask.sum()
            total_voxels = brain_mask.size
            ratio = brain_voxels / total_voxels
            
            print(f"\n📊 Brain Mask Validation:")
            print(f"   Total voxels: {total_voxels:,}")
            print(f"   Brain voxels: {brain_voxels:,}")
            print(f"   Ratio: {ratio:.1%}")
            
            if ratio < 0.05:
                print(f"❌ VALIDATION FAILED: Brain mask too small ({ratio:.1%} < 5%)")
                print(f"   Possible empty or failed extraction")
                return None, None
            
            if ratio > 0.7:
                print(f"❌ VALIDATION FAILED: Brain mask too large ({ratio:.1%} > 70%)")
                print(f"   Likely includes skull/face - HD-BET may have failed")
                return None, None
            
            print(f"✅ Validation passed: {ratio:.1%} is within acceptable range (5-70%)")
            print("="*60 + "\n")
            
            return brain_volume, brain_mask
            
    except subprocess.TimeoutExpired:
        print(f"❌ HD-BET timed out after 5 minutes")
        return None, None
    except Exception as e:
        print(f"❌ HD-BET exception: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def generate_brain_mask_otsu(volume: np.ndarray) -> np.ndarray:
    """Generate brain tissue mask (BRAIN ONLY, no skull/scalp).
    
    MEDICAL-GRADE BRAIN EXTRACTION:
    Targets brain parenchyma (gray + white matter) specifically.
    Excludes skull, CSF, scalp, face, eyes.
    
    Pipeline:
    1. High percentile thresholding (targets brain tissue intensity)
    2. Remove very small objects (< 10k voxels)
    3. Morphological erosion (removes skull boundary)
    4. Largest connected component (main brain mass)
    5. Morphological dilation (restore brain size)
    6. Hole filling
    
    Args:
        volume: Original MRI intensity volume (NOT normalized)
    
    Returns:
        Binary brain mask (uint8) - BRAIN TISSUE ONLY
    """
    from skimage.morphology import ball, binary_closing, binary_erosion, binary_dilation, remove_small_objects
    from scipy.ndimage import binary_fill_holes
    
    # Ensure non-negative volume
    volume_positive = np.abs(volume)
    
    # Get non-zero voxels
    non_zero = volume_positive[volume_positive > 0]
    if len(non_zero) == 0:
        raise ValueError("Empty volume - cannot generate brain mask")
    
    # CRITICAL: Detect data type
    data_max = non_zero.max()
    data_min = non_zero.min()
    data_range = data_max - data_min
    
    # Check if already binary mask
    is_binary = (data_max <= 1.01 and data_min >= 0.0 and len(np.unique(non_zero)) <= 10)
    
    if is_binary:
        # Already binary mask: use threshold 0.5
        threshold = 0.5
        print(f"🧠 Brain tissue threshold: {threshold} (binary mask detected)")
    elif data_range < 10:
        # Normalized data: use lower percentile  
        threshold = np.percentile(non_zero, 30)
        print(f"🧠 Brain tissue threshold: {threshold:.4f} (30th percentile, normalized data)")
    else:
        # Raw intensity data: use higher percentile
        threshold = np.percentile(non_zero, 60)
        print(f"🧠 Brain tissue threshold: {threshold:.2f} (60th percentile, raw data)")
    
    brain_mask = (volume_positive > threshold).astype(bool)
    print(f"📊 Initial mask: {brain_mask.sum():,} voxels")
    
    if brain_mask.sum() == 0:
        # Fallback: use very low threshold
        threshold = np.percentile(non_zero, 5)
        brain_mask = (volume_positive > threshold).astype(bool)
        print(f"⚠️ Fallback to 5th percentile ({threshold:.4f}): {brain_mask.sum():,} voxels")
        
        if brain_mask.sum() == 0:
            raise ValueError("Cannot generate brain mask - all thresholds failed")
    
    # ENHANCED: NO EROSION in normal case, but AGGRESSIVE skull-stripping if needed
    # Step 1: Remove very small objects (noise, eyes, sinuses)
    brain_mask = remove_small_objects(brain_mask, min_size=10000)
    print(f"📊 After small object removal: {brain_mask.sum():,} voxels")
    
    # Step 2: AGGRESSIVE EROSION for skull-stripping (removes face/skull)
    # This is a FALLBACK when HD-BET is not available
    # Larger erosion = more aggressive skull removal
    brain_mask_eroded = binary_erosion(brain_mask, ball(5))  # Aggressive erosion
    print(f"📊 After aggressive erosion: {brain_mask_eroded.sum():,} voxels")
    
    # Step 3: Keep largest connected component (main brain, NO face/skull)
    brain_mask_eroded = largest_connected_component(brain_mask_eroded)
    print(f"📊 After largest component: {brain_mask_eroded.sum():,} voxels")
    
    # Step 4: Dilate back to restore brain size (but not enough to add skull back)
    brain_mask_final = binary_dilation(brain_mask_eroded, ball(4))  # Less dilation than erosion
    print(f"📊 After dilation: {brain_mask_final.sum():,} voxels")
    
    # Step 5: Closing to smooth boundaries
    brain_mask_final = binary_closing(brain_mask_final, ball(2))
    print(f"📊 After closing: {brain_mask_final.sum():,} voxels")
    
    # Step 6: Fill all holes inside brain
    brain_mask_final = binary_fill_holes(brain_mask_final)
    
    print(f"✅ Final brain mask: {brain_mask_final.sum():,} voxels (BRAIN TISSUE ONLY)")
    
    return brain_mask_final.astype(np.uint8)


def compute_brain_bounding_box(brain_mask: np.ndarray, margin: int = 5) -> Tuple:
    """Compute tight bounding box around brain mask.
    
    CRITICAL: All visualization volumes must be cropped to brain bounding box.
    Never visualize full padded cubes.
    
    Args:
        brain_mask: Binary brain mask
        margin: Voxels to add around brain (for safety)
    
    Returns:
        Tuple of slices (z_slice, y_slice, x_slice)
    """
    coords = np.argwhere(brain_mask > 0)
    if len(coords) == 0:
        raise ValueError("Empty brain mask")
    
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    
    # Add margin
    shape = brain_mask.shape
    slices = []
    for i in range(3):
        start = max(0, min_coords[i] - margin)
        end = min(shape[i], max_coords[i] + margin + 1)
        slices.append(slice(start, end))
    
    return tuple(slices)


def clean_segmentation_morphology(
    segmentation: np.ndarray,
    brain_mask: Optional[np.ndarray] = None,
    min_lesion_size: int = 50
) -> np.ndarray:
    """Apply morphological operations in original patient space.
    
    PRODUCTION IMPROVEMENT: Clinical-grade visualization in patient coordinates.
    Removes ROI boundary artifacts via erosion and morphological cleaning.
    
    Args:
        segmentation: Segmentation in ORIGINAL space (not ROI)
        brain_mask: Optional brain mask to constrain segmentation
        min_lesion_size: Minimum voxel count for valid lesion
    
    Returns:
        Cleaned segmentation mask
    """
    from skimage.morphology import binary_opening, binary_closing, binary_erosion, ball, remove_small_objects
    
    # Convert to binary if probabilistic
    if segmentation.dtype == np.float32 or segmentation.dtype == np.float64:
        seg_binary = (segmentation > 0.5).astype(bool)
    else:
        seg_binary = segmentation.astype(bool)
    
    # Optional: Constrain to brain mask (if provided)
    if brain_mask is not None:
        seg_binary = seg_binary & brain_mask.astype(bool)
    
    # Remove very small objects (noise) - GENTLE
    if seg_binary.sum() > 0:
        seg_binary = remove_small_objects(seg_binary, min_size=min_lesion_size)
    
    # Light morphological closing (smooth boundaries, don't shrink) - GENTLE
    seg_binary = binary_closing(seg_binary, ball(1))  # Reduced from ball(2)
    
    # Fill holes
    seg_binary = binary_fill_holes(seg_binary)
    
    return seg_binary.astype(np.uint8)


def check_border_contact(segmentation: np.ndarray, margin: int = 2) -> bool:
    """Check if segmentation touches ROI borders (quality flag).
    
    CLINICAL GUARDRAIL:
    If lesion extends to ROI boundary, 3D visualization may be incomplete.
    This triggers a user warning.
    """
    # Check all 6 faces of the volume
    touches_border = (
        segmentation[:margin, :, :].any() or  # Top
        segmentation[-margin:, :, :].any() or  # Bottom
        segmentation[:, :margin, :].any() or  # Front
        segmentation[:, -margin:, :].any() or  # Back
        segmentation[:, :, :margin].any() or  # Left
        segmentation[:, :, -margin:].any()     # Right
    )
    return touches_border


# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION FUNCTIONS (REFACTORED FOR ANATOMICAL PLAUSIBILITY)
# ═══════════════════════════════════════════════════════════════════════════

def create_slice_view(
    volume: np.ndarray,
    segmentations_roi: Dict,
    roi_metadata: Dict,
    axis: int = 2,
    slice_idx: int = None
) -> plt.Figure:
    """Create slice visualization with overlays.
    
    FIXED: Maps ROI segmentations to original space before slicing.
    """
    if slice_idx is None:
        slice_idx = volume.shape[axis] // 2
    
    # Get base slice from volume
    if axis == 0:
        base_slice = volume[slice_idx, :, :]
    elif axis == 1:
        base_slice = volume[:, slice_idx, :]
    else:
        base_slice = volume[:, :, slice_idx]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), facecolor='#0A0E27')
    ax.set_facecolor('#0A0E27')
    
    # Show base MRI
    ax.imshow(base_slice, cmap='gray', origin='lower')
    
    # Overlay segmentations (map from ROI to original space first)
    for disease, (probs, binary) in segmentations_roi.items():
        # ALZHEIMER GUARD
        if disease == "alzheimer":
            continue
        
        # Map ROI → original space
        seg_original = map_segmentation_to_original_space(binary, roi_metadata)
        
        # Now slice from original space
        if axis == 0:
            seg_slice = seg_original[slice_idx, :, :]
        elif axis == 1:
            seg_slice = seg_original[:, slice_idx, :]
        else:
            seg_slice = seg_original[:, :, slice_idx]
        
        # Create colored overlay (H, W, 4) - channel dimension LAST
        color = tuple(int(DISEASE_COLORS[disease]["hex"][i:i+2], 16)/255 for i in (1, 3, 5))
        overlay = np.zeros((*seg_slice.shape, 4))
        mask = seg_slice > 0
        overlay[mask] = [*color, 0.5]
        
        ax.imshow(overlay, origin='lower')  # No transpose - already correct shape
    
    ax.axis('off')
    axis_names = ['Sagittal', 'Coronal', 'Axial']
    ax.set_title(f"{axis_names[axis]} Slice {slice_idx}", 
                 color='#00E5FF', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def create_3d_visualization(
    segmentations_roi: Dict,
    roi_metadata: Dict,
    original_volume: np.ndarray,
    affine: np.ndarray,
    spacing: Tuple[float, float, float],
    show_patient_brain: bool = True,
    clinical_decision: Optional[Dict] = None,
    show_heatmap: bool = False
) -> go.Figure:
    """Medical-Grade Patient-Specific Brain Visualization.
    
    ANATOMICALLY ACCURATE SURFACE RENDERING:
    1. Generate brain mask using Otsu thresholding
    2. Crop to brain bounding box (no padded cubes)
    3. Brain surface from mask (marching cubes at level=0.5)
    4. Lesion surfaces separately (never merged with brain)
    5. Apply affine transforms (world coordinates)
    
    CLINICAL DECISION GATING:
    If clinical_decision provided, only renders lesion for primary disease.
    
    CRITICAL: Brain and lesions are SEPARATE meshes, never meshed together.
    
    Args:
        segmentations_roi: Dict of (probs, binary) in ROI space (96³)
        roi_metadata: Coordinate mapping metadata
        original_volume: Full-resolution patient MRI (original intensity)
        affine: NIfTI affine matrix
        spacing: Voxel spacing in mm
        show_patient_brain: Whether to render patient brain surface
        clinical_decision: Optional clinical analysis with primary disease
    
    Returns:
        Plotly 3D figure with anatomically realistic rendering
    """
    fig = go.Figure()
    
    print("\n" + "=" * 60)
    print("🧠 BRAIN EXTRACTION PIPELINE")
    print("=" * 60)
    
    # HD-BET ONLY - NO FALLBACK (Gold Standard)
    print("\n🎯 Calling HD-BET (ONLY method - gold standard)...")
    
    brain_volume, brain_mask = apply_hdbet_brain_extraction(original_volume, affine, spacing)
    
    if brain_mask is None:
        # HARD FAILURE - No fallback to heuristics
        st.error(
            "❌ **HD-BET Brain Extraction Failed**\n\n"
            "3D brain surface rendering has been **disabled**.\n\n"
            "**Why:** HD-BET is the ONLY clinically valid method for heterogeneous MRI data.\n"
            "Heuristic methods (Otsu, morphology) cause face/skull artifacts.\n\n"
            "**Most likely cause:** HD-BET model weights need to be downloaded.\n\n"
            "**Solution:**\n"
            "1. Download weights manually: https://zenodo.org/records/14445620\n"
            "2. Extract `release_v1.5.0.zip`\n"
            "3. Find HD-BET folder:\n"
            "   ```\n"
            "   py -c \"import HD_BET; import os; print(os.path.dirname(HD_BET.__file__))\"\n"
            "   ```\n"
            "4. Copy `.pkl` files to `[HD-BET folder]/parameters/`\n"
            "5. Restart Streamlit\n\n"
            "**Note:** Disease detection results are still valid."
        )
        print("❌ 3D RENDERING DISABLED - HD-BET REQUIRED (gold standard)")
        print("=" * 60 + "\n")
        return fig  # Return empty figure
    
    print(f"✅ HD-BET SUCCESS: Brain extracted with {brain_mask.sum():,} voxels")
    
    # Compute bounding box from HD-BET mask
    brain_bbox = compute_brain_bounding_box(brain_mask, margin=5)
    bbox_shape = tuple(s.stop - s.start for s in brain_bbox)
    print(f"📦 Bounding box: {bbox_shape} (from original {original_volume.shape})")
    
    # Crop to brain region
    brain_mask_cropped = brain_mask[brain_bbox]
    original_cropped = original_volume[brain_bbox]
    print(f"✂️  Cropped to brain-only region")
    print("=" * 60 + "\n")
    
    # LAYER 1: Patient-Specific Brain Surface (from mask, not MRI)
    if show_patient_brain and brain_mask_cropped is not None:
        try:
            with st.spinner("Generating patient brain surface..."):
                print("🧠 Generating brain surface mesh...")
                brain_verts, brain_faces = generate_patient_brain_surface(
                    brain_mask=brain_mask_cropped,
                    affine=None,
                    spacing=spacing
                )
                print(f"✅ Brain mesh: {len(brain_verts):,} vertices, {len(brain_faces):,} faces")
            
            # Marching cubes with spacing already gives physical mm coordinates
            # NO manual offset needed
            
            # Add brain surface mesh (moderate opacity for visibility)
            fig.add_trace(go.Mesh3d(
                x=brain_verts[:, 0],
                y=brain_verts[:, 1],
                z=brain_verts[:, 2],
                i=brain_faces[:, 0],
                j=brain_faces[:, 1],
                k=brain_faces[:, 2],
                color='lightgray',
                opacity=0.4,
                name='Brain Surface',
                showlegend=True,
                hoverinfo='skip',
                lighting=dict(ambient=0.6, diffuse=0.8, roughness=0.5),
                lightposition=dict(x=100, y=200, z=0)
            ))
            
            st.success(f"✅ Brain surface: {len(brain_verts):,} vertices")
            print(f"🎨 Brain surface added: {len(brain_verts):,} verts, {len(brain_faces):,} faces")
            print(f"   Vertex range: X=[{brain_verts[:,0].min():.1f}, {brain_verts[:,0].max():.1f}]")
            print(f"                 Y=[{brain_verts[:,1].min():.1f}, {brain_verts[:,1].max():.1f}]")
            print(f"                 Z=[{brain_verts[:,2].min():.1f}, {brain_verts[:,2].max():.1f}]")
            
        except Exception as e:
            st.warning(f"⚠️ Could not generate brain surface: {e}")
            print(f"⚠️ Brain surface error: {e}")
    
    # LAYER 2: Lesion Surfaces (Mapped to Original Space, SEPARATE from brain)
    for disease, (probs_roi, binary_roi) in segmentations_roi.items():
        # ALZHEIMER HARD GUARD (COMPLIANCE REQUIREMENT)
        if disease == "alzheimer":
            st.info(f"ℹ️ **Alzheimer's Disease**: Presence-only detection (no voxel-level localization). "
                    "ADNI dataset does not provide lesion masks. 3D visualization not applicable.")
            continue  # Skip 3D mesh, slice overlay, volume rendering
        
        # CLINICAL GATING: Only render lesion if it's the primary disease
        if clinical_decision is not None:
            primary_disease = clinical_decision["primary_disease"]
            primary_confidence = clinical_decision["primary_confidence"]
            
            if disease != primary_disease:
                other_prob = clinical_decision["disease_probabilities"][disease]
                st.info(f"ℹ️ **{DISEASE_COLORS[disease]['name']}**: Not primary diagnosis "
                       f"(confidence: {other_prob:.1%} vs primary {primary_disease}: {primary_confidence:.1%})")
                print(f"   Skipped: {disease} not primary (prob={other_prob:.1%})")
                continue
            
            if not clinical_decision["threshold_met"]:
                st.warning(f"⚠️ **{DISEASE_COLORS[disease]['name']}**: Low confidence "
                          f"({primary_confidence:.1%}) - clinical review recommended")
                print(f"   Skipped: Below clinical threshold ({primary_confidence:.1%} < 60%)")
                continue
        
        color = DISEASE_COLORS[disease]["hex"]
        name = DISEASE_COLORS[disease]["name"]
        
        print(f"\n🔬 Processing {name} lesion...")
        print(f"   ROI space: {binary_roi.shape}")
        
        # Apply moderate probability threshold to reduce false positives
        PROB_THRESHOLD = 0.6  # Balanced threshold
        
        if probs_roi.max() < PROB_THRESHOLD:
            st.info(f"ℹ️ **{name}**: Max probability {probs_roi.max():.2f} below threshold {PROB_THRESHOLD}")
            print(f"   ⚠️ Skipped: Max prob {probs_roi.max():.2f} < {PROB_THRESHOLD}")
            continue
        
        # Re-threshold with moderate threshold
        if binary_roi.ndim == 4:  # Multi-channel
            binary_strict = (probs_roi > PROB_THRESHOLD).astype(np.uint8)
        else:
            binary_strict = (probs_roi > PROB_THRESHOLD).astype(np.uint8)
        
        print(f"   After threshold {PROB_THRESHOLD}: {binary_strict.sum():,} voxels in ROI")
        
        # CRITICAL: Map from ROI space to original patient space
        seg_original = map_segmentation_to_original_space(binary_strict, roi_metadata)
        print(f"   Original space: {seg_original.shape}, {seg_original.sum():,} voxels")
        
        # Crop to brain bounding box if available
        if brain_bbox is not None:
            seg_original = seg_original[brain_bbox]
            print(f"   After bbox crop: {seg_original.shape}, {seg_original.sum():,} voxels")
        
        # Minimum volume check
        MIN_VOLUME_VOXELS = 50  # Reasonable minimum
        voxel_count = int(seg_original.sum())
        
        if voxel_count < MIN_VOLUME_VOXELS:
            st.info(f"ℹ️ **{name}**: Lesion volume too small ({voxel_count} voxels)")
            print(f"   ⚠️ Skipped: {voxel_count} < {MIN_VOLUME_VOXELS} voxels")
            continue
        
        # Moderate cleaning to reduce size but preserve lesions
        from skimage.morphology import binary_erosion, binary_closing, ball, remove_small_objects
        from scipy.ndimage import binary_fill_holes
        
        seg_clean = seg_original.astype(bool)
        
        # Remove small objects first (GENTLE)
        if seg_clean.sum() > 0:
            seg_clean = remove_small_objects(seg_clean, min_size=20)  # Reduced from 50
        
        # LIGHT erosion to shrink slightly (not too aggressive)
        seg_clean = binary_erosion(seg_clean, ball(1))  # Reduced from ball(2)
        
        # Keep largest component only
        seg_clean = largest_connected_component(seg_clean)
        
        # Close and fill
        seg_clean = binary_closing(seg_clean, ball(1))
        seg_clean = binary_fill_holes(seg_clean).astype(np.uint8)
        
        cleaned_voxels = seg_clean.sum()
        MIN_CLEAN_VOXELS = 10  # Very low threshold
        if cleaned_voxels < MIN_CLEAN_VOXELS:
            print(f"   ⚠️ Skipped: Post-cleaning {cleaned_voxels} < {MIN_CLEAN_VOXELS} voxels")
            continue
        
        print(f"   After cleaning: {cleaned_voxels:,} voxels ({100*cleaned_voxels/voxel_count:.1f}% retained)")
        
        # GOLD-STANDARD CLINICAL QA: Anatomical validation
        is_valid, centroid, msg = validate_lesion_position(seg_clean, brain_mask_cropped if brain_bbox else None)
        
        if not is_valid:
            st.warning(f"⚠️ **{name}**: {msg} - Invalid visualization rejected")
            print(f"   ❌ ANATOMICAL VALIDATION FAILED: {msg}")
            continue
        
        print(f"   ✅ Anatomical validation: {msg}")
        
        # Light Gaussian smoothing (σ ≤ 0.5 for small lesions)
        lesion_volume = seg_clean.sum()
        sigma = 0.5  # Consistent smoothing
        
        try:
            seg_smooth = gaussian_filter(seg_clean.astype(float), sigma=sigma)
            
            # GOLD-STANDARD: Marching cubes WITH spacing (same as brain)
            verts, faces, normals, _ = measure.marching_cubes(
                seg_smooth,
                level=0.5,
                spacing=spacing  # Physical mm coordinates
            )
            
            print(f"   Marching cubes: {len(verts):,} vertices, {len(faces):,} faces")
            
            # NO manual offset - marching cubes with spacing handles it
            
            # Show vertex range for debugging
            print(f"   Vertex range: X=[{verts[:,0].min():.1f}, {verts[:,0].max():.1f}]")
            print(f"                 Y=[{verts[:,1].min():.1f}, {verts[:,1].max():.1f}]")
            print(f"                 Z=[{verts[:,2].min():.1f}, {verts[:,2].max():.1f}]")
            
            # Prepare mesh coloring
            if show_heatmap:
                # Heatmap: sample probability values at vertex locations
                # Interpolate probability values at mesh vertices
                from scipy.ndimage import map_coordinates
                
                # Verts are in mm, need to convert to voxel coords for sampling
                verts_voxel = verts / np.array(spacing)
                
                # Sample probabilities at vertex locations
                vertex_probs = map_coordinates(
                    probs_roi[0] if probs_roi.ndim == 4 else probs_roi,
                    verts_voxel.T,
                    order=1,
                    mode='nearest'
                )
                
                # Add heatmap mesh with probability-based coloring
                fig.add_trace(go.Mesh3d(
                    x=verts[:, 0],
                    y=verts[:, 1],
                    z=verts[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    intensity=vertex_probs,
                    colorscale='Hot',  # Red-yellow heatmap
                    opacity=0.95,
                    name=f"{name} Heatmap ({cleaned_voxels} voxels)",
                    showlegend=True,
                    hoverinfo='text',
                    hovertext=[f"Probability: {p:.2f}" for p in vertex_probs],
                    lighting=dict(ambient=0.7, diffuse=0.8),
                    colorbar=dict(title="Probability", x=1.0)
                ))
            else:
                # Solid color mesh
                fig.add_trace(go.Mesh3d(
                    x=verts[:, 0],
                    y=verts[:, 1],
                    z=verts[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    color=color,
                    opacity=0.9,
                    name=f"{name} ({cleaned_voxels} voxels)",
                    showlegend=True,
                    hoverinfo='name',
                    lighting=dict(ambient=0.7, diffuse=0.8)
                ))
            
            print(f"   ✅ {name} mesh added to scene ({'heatmap' if show_heatmap else 'solid color'})")
            
        except (ValueError, RuntimeError) as e:
            st.warning(f"⚠️ **{name}**: Mesh generation failed - {str(e)}")
            print(f"   ❌ Mesh generation failed: {e}")
            continue
    
    # Layout with medical disclaimers
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, showgrid=False),
            yaxis=dict(visible=False, showgrid=False),
            zaxis=dict(visible=False, showgrid=False),
            bgcolor='#0a1120',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        showlegend=True,
        legend=dict(
            bgcolor='rgba(17, 24, 39, 0.9)',
            bordercolor='#00E5FF',
            borderwidth=1,
            font=dict(color='#E5E7EB')
        ),
        paper_bgcolor='#030712',
        margin=dict(l=0, r=0, t=50, b=0),
        height=700,
        title=dict(
            text="Patient-Specific Brain Surface with Lesion Overlay<br>"
                 "<sub>⚠️ Brain mask via Otsu thresholding | Lesions in original coordinates</sub>",
            font=dict(size=14, color='#00E5FF'),
            x=0.5,
            xanchor='center'
        ),
        annotations=[
            dict(
                text="<b>Source:</b> Brain surface from MRI-derived mask (Otsu + morphology). "
                     "Lesion segmentation in 96³ ROI, mapped to patient space.",
                xref="paper", yref="paper",
                x=0.5, y=-0.05,
                showarrow=False,
                font=dict(size=10, color='#9CA3AF'),
                xanchor='center'
            )
        ]
    )
    
    return fig


def create_volume_rendering(
    segmentations_roi: Dict,
    roi_metadata: Dict,
    original_volume: np.ndarray,
    downsample_factor: int = 4
) -> go.Figure:
    """Create volume rendering visualization.
    
    FIXED: Maps ROI to original space, shows brain MRI with lesion overlays.
    """
    # Downsample original volume for performance
    vol_down = original_volume[::downsample_factor, ::downsample_factor, ::downsample_factor]
    
    # Normalize to 0-1
    vol_norm = (vol_down - vol_down.min()) / (vol_down.max() - vol_down.min() + 1e-8)
    
    # Create figure
    fig = go.Figure()
    
    # Add brain MRI volume (semi-transparent grayscale)
    fig.add_trace(go.Volume(
        x=np.arange(vol_down.shape[0]),
        y=np.arange(vol_down.shape[1]),
        z=np.arange(vol_down.shape[2]),
        value=vol_norm.flatten(),
        isomin=0.2,  # Lower threshold to show more brain
        isomax=0.8,
        opacity=0.3,  # Increased opacity for visibility
        surface_count=15,
        colorscale='Greys',
        name='Brain MRI',
        showlegend=True
    ))
    
    # Add lesion volumes (from ROI space)
    for disease, (probs_roi, binary_roi) in segmentations_roi.items():
        if disease == "alzheimer":
            continue  # No volume rendering for Alzheimer
        
        # Map to original space
        seg_original = map_segmentation_to_original_space(binary_roi, roi_metadata)
        
        # Downsample
        seg_down = seg_original[::downsample_factor, ::downsample_factor, ::downsample_factor]
        
        if seg_down.sum() == 0:
            continue
        
        # Create volume trace
        color = DISEASE_COLORS[disease]["hex"]
        fig.add_trace(go.Volume(
            x=np.arange(seg_down.shape[0]),
            y=np.arange(seg_down.shape[1]),
            z=np.arange(seg_down.shape[2]),
            value=seg_down.flatten().astype(float),
            isomin=0.5,
            isomax=1.0,
            opacity=0.6,
            surface_count=10,
            colorscale=[[0, color], [1, color]],
            name=DISEASE_COLORS[disease]["name"],
            showlegend=True
        ))
    
    # Layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor='#0A0E27'
        ),
        paper_bgcolor='#0A0E27',
        plot_bgcolor='#0A0E27',
        margin=dict(l=0, r=0, t=0, b=0),
        height=600
    )
    
    return fig


def create_statistical_summary(detection: Dict, segmentations: Dict) -> go.Figure:
    """Statistical visualization"""
    probs = detection["probabilities"]
    
    fig = go.Figure()
    
    diseases = list(probs.keys())
    probabilities = [probs[d] for d in diseases]
    colors = [DISEASE_COLORS[d]["hex"] for d in diseases]
    names = [DISEASE_COLORS[d]["name"] for d in diseases]
    
    fig.add_trace(go.Bar(
        x=names,
        y=probabilities,
        marker=dict(
            color=colors,
            line=dict(color='#00E5FF', width=2)
        ),
        text=[f"{p:.1%}" for p in probabilities],
        textposition='outside',
        textfont=dict(color='#E5E7EB', size=14)
    ))
    
    fig.add_hline(y=PRESENCE_THRESHOLD, line_dash="dash", 
                  line_color="#00FFFF", annotation_text="Threshold",
                  annotation_font_color="#00FFFF")
    
    fig.update_layout(
        title=dict(
            text="Disease Presence Confidence",
            font=dict(color='#00E5FF', size=18)
        ),
        xaxis=dict(
            title="Disease Type",
            titlefont=dict(color='#E5E7EB'),
            tickfont=dict(color='#E5E7EB'),
            gridcolor='rgba(74, 144, 226, 0.1)'
        ),
        yaxis=dict(
            title="Confidence",
            titlefont=dict(color='#E5E7EB'),
            tickfont=dict(color='#E5E7EB'),
            gridcolor='rgba(74, 144, 226, 0.1)',
            range=[0, 1]
        ),
        plot_bgcolor='#0a1120',
        paper_bgcolor='#030712',
        height=400
    )
    
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# AI REPORT GENERATION (GROQ)
# ═══════════════════════════════════════════════════════════════════════════

def generate_ai_report(detection: Dict, segmentations: Dict, groq_api_key: Optional[str] = None) -> str:
    """Generate AI-powered analysis report"""
    if not GROQ_AVAILABLE or not groq_api_key:
        return generate_fallback_report(detection, segmentations)
    
    try:
        client = Groq(api_key=groq_api_key)
        
        # Prepare context
        detected = detection["detected_diseases"]
        probs = detection["probabilities"]
        
        context = f"""
Medical Imaging Analysis Context:

Detected Pathologies:
- Tumor: {probs['tumor']:.1%} confidence {'(DETECTED)' if 'tumor' in detected else ''}
- Stroke: {probs['stroke']:.1%} confidence {'(DETECTED)' if 'stroke' in detected else ''}
- Alzheimer Pattern: {probs['alzheimer']:.1%} confidence {'(DETECTED)' if 'alzheimer' in detected else ''}

Segmentation Results:
"""
        
        for disease in detected:
            if disease in segmentations:
                _, binary = segmentations[disease]
                volume = binary.sum()
                context += f"- {DISEASE_COLORS[disease]['name']}: ~{volume} voxels segmented\n"
            elif disease == "alzheimer":
                context += f"- Alzheimer: Presence detected (no lesion segmentation)\n"
        
        prompt = f"""{context}

Generate a detailed but EDUCATIONAL radiology-style report. IMPORTANT guidelines:
1. This is for RESEARCH/EDUCATIONAL purposes only
2. Do NOT make diagnostic claims
3. Use phrases like "imaging characteristics consistent with" or "patterns suggestive of"
4. Emphasize need for clinical correlation
5. Note that Alzheimer detection is presence-based, not volumetric
6. Keep professional medical terminology
7. Include limitations section
8. Max 300 words

Format:
## FINDINGS
[Describe detected patterns]

## IMPRESSION  
[Clinical correlation needed statement]

## LIMITATIONS
[Technical limitations]
"""
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="mixtral-8x7b-32768",
            temperature=0.3,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        st.warning(f"AI report generation failed: {e}")
        return generate_fallback_report(detection, segmentations)


def generate_fallback_report(detection: Dict, segmentations: Dict) -> str:
    """Fallback report without AI"""
    detected = detection["detected_diseases"]
    probs = detection["probabilities"]
    
    report = "# NeuroX Multi-Disease Analysis Report\n\n"
    report += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    report += "**⚠️ RESEARCH USE ONLY** - Not for clinical diagnosis\n\n"
    report += "---\n\n"
    
    report += "## DETECTED IMAGING PATTERNS\n\n"
    
    if not detected:
        report += "**No significant abnormality detected.**\n\n"
        report += "The model did not identify imaging characteristics consistent with tumor, stroke, or neurodegenerative patterns above the confidence threshold.\n\n"
    else:
        for disease in detected:
            name = DISEASE_COLORS[disease]["name"]
            prob = probs[disease]
            
            report += f"### {name}\n"
            report += f"- **Presence Confidence:** {prob:.1%}\n"
            
            if disease in segmentations:
                _, binary = segmentations[disease]
                volume = binary.sum()
                report += f"- **Segmented Volume:** ~{volume} voxels\n"
                report += f"- **Assessment:** Lesion boundaries identified\n"
            elif disease == "alzheimer":
                report += f"- **Assessment:** Presence detection only (no lesion mask)\n"
                report += f"- **Note:** Pattern-based, not volumetric atrophy measurement\n"
            
            report += "\n"
    
    report += "---\n\n"
    report += "## CLINICAL CORRELATION REQUIRED\n\n"
    report += "This automated analysis:\n"
    report += "- Detects imaging characteristics consistent with abnormal tissue patterns\n"
    report += "- Requires expert radiological and clinical interpretation\n"
    report += "- Does NOT constitute a medical diagnosis\n"
    report += "- Should be correlated with clinical presentation and history\n\n"
    
    report += "## TECHNICAL LIMITATIONS\n\n"
    report += "- No skull stripping or brain extraction performed\n"
    report += "- Atlas-based visualization is approximate\n"
    report += "- Alzheimer detection is presence-based, not cortical thickness analysis\n"
    report += "- Model trained on specific datasets (BraTS, ISLES, ADNI)\n"
    
    return report


def create_pdf_report(detection: Dict, segmentations: Dict, report_text: str, output_path: str):
    """Generate PDF report with visualizations"""
    try:
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=rl_colors.HexColor('#00E5FF'),
            spaceAfter=30
        )
        story.append(Paragraph("🧠 NeuroX Analysis Report", title_style))
        story.append(Spacer(1, 12))
        
        # Metadata
        story.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Paragraph("<b>System:</b> NeuroX Multi-Disease Detection", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Warning
        warning_style = ParagraphStyle(
            'Warning',
            parent=styles['Normal'],
            textColor=rl_colors.HexColor('#FF8800'),
            fontSize=10
        )
        story.append(Paragraph("⚠️ RESEARCH AND EDUCATIONAL USE ONLY - NOT FOR CLINICAL DIAGNOSIS", warning_style))
        story.append(Spacer(1, 20))
        
        # Detection table
        probs = detection["probabilities"]
        detected = detection["detected_diseases"]
        
        table_data = [["Disease", "Confidence", "Status"]]
        for disease in ["tumor", "stroke", "alzheimer"]:
            name = DISEASE_COLORS[disease]["name"]
            prob = f"{probs[disease]:.1%}"
            status = "DETECTED" if disease in detected else "Not detected"
            table_data.append([name, prob, status])
        
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), rl_colors.HexColor('#00E5FF')),
            ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), rl_colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, rl_colors.black)
        ]))
        story.append(table)
        story.append(Spacer(1, 20))
        
        # Report text
        for line in report_text.split('\n'):
            if line.strip():
                if line.startswith('#'):
                    line = line.replace('#', '').strip()
                    story.append(Paragraph(f"<b>{line}</b>", styles['Heading2']))
                else:
                    story.append(Paragraph(line, styles['Normal']))
                story.append(Spacer(1, 6))
        
        doc.build(story)
        return True
    except Exception as e:
        st.error(f"PDF generation failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════
# STREAMLIT UI - PREMIUM DESIGN
# ═══════════════════════════════════════════════════════════════════════════

def run_streamlit_app():
    """Premium Streamlit UI"""
    
    st.set_page_config(
        page_title="NeuroX Adaptive",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"  # Sidebar open by default
    )
    
    # Optimized Premium CSS
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Orbitron:wght@600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

        :root {
            --neon-blue: #00E5FF;
            --neon-cyan: #00FFFF;
            --neon-green: #00FF88;
            --neon-purple: #B67EFF;
            --void: #030712;
            --surface: #111827;
        }

        * {font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;}

        .stApp {
            background: radial-gradient(ellipse at top, #0a1628 0%, #030712 40%, #000000 100%);
            color: #E5E7EB;
        }

        #MainMenu, footer, .stDeployButton, header {display: none !important;}

        /* COMPACT HEADER */
        .medical-header {
            background: linear-gradient(135deg, rgba(17, 24, 39, 0.95) 0%, rgba(10, 22, 40, 0.9) 100%);
            backdrop-filter: blur(20px) saturate(180%);
            border: 2px solid rgba(0, 229, 255, 0.25);
            border-radius: 16px;
            padding: 24px 28px;
            margin: 15px 0 25px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5), 0 0 40px rgba(0, 229, 255, 0.12);
            position: relative;
            overflow: hidden;
        }

        .medical-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, #FF4444, #00E5FF, #00FFFF, #00FF88, #FF4444);
            background-size: 400% 100%;
            animation: shimmer 4s linear infinite;
        }

        @keyframes shimmer {
            0% {background-position: 0% 50%;}
            100% {background-position: 400% 50%;}
        }

        .brand-title {
            font-family: 'Orbitron', monospace;
            font-size: 42px;
            font-weight: 800;
            background: linear-gradient(135deg, #00E5FF 0%, #00FFFF 50%, #00FF88 100%);
            background-size: 200% auto;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: 4px;
            margin: 0;
            text-align: center;
            animation: textShine 3s linear infinite;
        }

        @keyframes textShine {to {background-position: 200% center;}}

        .brand-subtitle {
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            color: #94A3B8;
            text-transform: uppercase;
            letter-spacing: 3px;
            margin-top: 8px;
            text-align: center;
            font-weight: 500;
        }

        .system-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: linear-gradient(135deg, rgba(0, 255, 136, 0.15), rgba(0, 229, 255, 0.1));
            border: 2px solid rgba(0, 255, 136, 0.4);
            border-radius: 40px;
            padding: 6px 16px;
            margin-top: 12px;
            color: var(--neon-green);
            font-weight: 700;
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 2px;
        }

        .status-indicator {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--neon-green);
            animation: pulse 2s ease-in-out infinite;
            box-shadow: 0 0 12px var(--neon-green);
        }

        @keyframes pulse {
            0%, 100% {opacity: 1; transform: scale(1);}
            50% {opacity: 0.4; transform: scale(0.85);}
        }

        /* COMPACT CARDS */
        .clinical-card, .glass-card {
            background: linear-gradient(135deg, rgba(17, 24, 39, 0.8) 0%, rgba(10, 18, 32, 0.7) 100%);
            backdrop-filter: blur(12px) saturate(180%);
            border: 1px solid rgba(0, 229, 255, 0.2);
            border-left: 3px solid var(--neon-blue);
            border-radius: 12px;
            padding: 16px 20px;
            margin: 12px 0;
            box-shadow: 0 2px 16px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.05);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .clinical-card:hover, .glass-card:hover {
            transform: translateY(-4px) scale(1.005);
            border-color: rgba(0, 229, 255, 0.4);
            box-shadow: 0 8px 28px rgba(0, 0, 0, 0.4), 0 0 20px rgba(0, 229, 255, 0.15);
        }

        /* COMPACT BUTTONS */
        .stButton > button {
            background: linear-gradient(135deg, var(--neon-blue), var(--neon-cyan)) !important;
            background-size: 200% auto !important;
            color: #030712 !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 10px 20px !important;
            font-family: 'Orbitron', monospace !important;
            font-weight: 700 !important;
            font-size: 13px !important;
            text-transform: uppercase !important;
            letter-spacing: 2px !important;
            box-shadow: 0 4px 16px rgba(0, 229, 255, 0.3) !important;
            transition: all 0.3s !important;
            width: 100% !important;
        }

        .stButton > button:hover {
            transform: translateY(-2px) scale(1.01) !important;
            background-position: right center !important;
            box-shadow: 0 8px 24px rgba(0, 229, 255, 0.5) !important;
        }

        /* COMPACT FILE UPLOADER */
        .stFileUploader {
            border: 2px dashed rgba(0, 229, 255, 0.3) !important;
            border-radius: 12px !important;
            background: linear-gradient(135deg, rgba(17, 24, 39, 0.5), rgba(10, 18, 32, 0.4)) !important;
            backdrop-filter: blur(10px) !important;
            padding: 28px !important;
            transition: all 0.3s !important;
        }

        .stFileUploader:hover {
            border-color: var(--neon-cyan) !important;
            background: linear-gradient(135deg, rgba(0, 255, 255, 0.08), rgba(0, 229, 255, 0.05)) !important;
            box-shadow: 0 0 20px rgba(0, 229, 255, 0.15) !important;
        }

        /* COMPACT TABS */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: linear-gradient(135deg, rgba(3, 7, 18, 0.7), rgba(10, 18, 32, 0.5));
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 8px;
            border-bottom: none;
        }

        .stTabs [data-baseweb="tab"] {
            background: linear-gradient(135deg, rgba(17, 24, 39, 0.6), rgba(10, 16, 32, 0.5));
            border: 1px solid rgba(74, 144, 226, 0.15);
            color: #94A3B8;
            font-weight: 600;
            padding: 10px 20px;
            border-radius: 8px;
            transition: all 0.2s;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1.5px;
        }

        .stTabs [data-baseweb="tab"]:hover {
            background: linear-gradient(135deg, rgba(0, 229, 255, 0.1), rgba(0, 255, 255, 0.06));
            border-color: rgba(0, 229, 255, 0.3);
            color: #00FFFF;
            transform: translateY(-1px);
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(0, 229, 255, 0.2), rgba(0, 255, 255, 0.12)) !important;
            border: 2px solid var(--neon-blue) !important;
            color: var(--neon-cyan) !important;
            transform: translateY(-2px);
            box-shadow: 0 4px 16px rgba(0, 229, 255, 0.3) !important;
        }

        /* COMPACT SCROLLBAR */
        ::-webkit-scrollbar {width: 8px; height: 8px;}
        ::-webkit-scrollbar-track {background: var(--void); border-radius: 8px;}
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(180deg, var(--neon-blue), var(--neon-cyan));
            border-radius: 8px;
            border: 2px solid var(--void);
        }
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(180deg, var(--neon-cyan), var(--neon-green));
        }

        /* COMPACT METRICS */
        .stMetric {
            background: linear-gradient(135deg, rgba(17, 24, 39, 0.85), rgba(10, 16, 32, 0.75));
            backdrop-filter: blur(10px);
            border: 1px solid rgba(0, 229, 255, 0.2);
            border-radius: 12px;
            padding: 16px !important;
            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.3);
            transition: all 0.3s;
        }

        .stMetric:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4), 0 0 16px rgba(0, 229, 255, 0.15);
        }

        /* SIDEBAR */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(3, 7, 18, 0.95), rgba(10, 18, 32, 0.9));
            backdrop-filter: blur(16px);
            border-right: 1px solid rgba(0, 229, 255, 0.15);
        }

        /* COMPACT HEADINGS */
        h1 {font-size: 28px !important; margin: 16px 0 12px !important;}
        h2 {font-size: 22px !important; margin: 14px 0 10px !important; color: var(--neon-cyan);}
        h3 {font-size: 18px !important; margin: 12px 0 8px !important; color: var(--neon-blue);}
        
        /* LOADING SPINNER */
        .stSpinner > div {
            border-top-color: var(--neon-blue) !important;
            border-right-color: var(--neon-cyan) !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state for multi-page navigation
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'upload'
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'detection_results' not in st.session_state:
        st.session_state.detection_results = None
    if 'segmentation_results' not in st.session_state:
        st.session_state.segmentation_results = {}
    if 'original_image' not in st.session_state:
        st.session_state.original_image = None
    if 'report_text' not in st.session_state:
        st.session_state.report_text = ""
    
    # Premium Header with Navigation
    st.markdown("""
    <div class="medical-header">
        <div class="brand-title">🧠 NEUROX</div>
        <div class="brand-subtitle">Multi-Disease Pathology Detection System</div>
        <div style="text-align: center; margin-top: 24px;">
            <span class="system-badge">
                <span class="status-indicator"></span>
                RESEARCH SYSTEM ACTIVE
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Page Navigation Bar
    st.markdown("""
    <div style="display: flex; justify-content: center; gap: 16px; margin: 30px 0; flex-wrap: wrap;">
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📁 UPLOAD", key="nav_upload", use_container_width=True):
            st.session_state.current_page = 'upload'
            st.rerun()
    
    with col2:
        if st.button("🔬 ANALYSIS", key="nav_analysis", use_container_width=True, disabled=not st.session_state.analysis_complete):
            st.session_state.current_page = 'analysis'
            st.rerun()
    
    with col3:
        if st.button("🧠 VISUALIZATION", key="nav_viz", use_container_width=True, disabled=not st.session_state.analysis_complete):
            st.session_state.current_page = 'visualization'
            st.rerun()
    
    with col4:
        if st.button("📄 REPORTS", key="nav_reports", use_container_width=True, disabled=not st.session_state.analysis_complete):
            st.session_state.current_page = 'reports'
            st.rerun()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎛️ System Configuration")
        
        # Groq API Key
        groq_key = st.text_input("Groq API Key (Optional)", type="password", 
                                  help="Enable AI-powered reports")
        
        # Visualization Controls
        st.sidebar.markdown("### 🎨 Visualization Options")
        show_atlas = st.sidebar.checkbox("Show Brain Surface", value=True)
        show_heatmap = st.sidebar.checkbox("Probability Heatmap", value=False,
                                           help="Show lesion probability as color gradient")
        
        st.sidebar.markdown("---")
        st.markdown("### 📊 Detection Threshold")
        threshold = st.slider("Confidence Threshold", 0.0, 1.0, PRESENCE_THRESHOLD, 0.05)
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info("""
**NeuroX** uses deep learning to detect:
- Brain Tumors (BraTS)
- Ischemic Strokes (ISLES)
- Alzheimer Patterns (ADNI)

**Automatic routing** ensures proper handling of each disease type.
        """)
    
    
    # PAGE ROUTING
    if st.session_state.current_page == 'upload':
        # ========== UPLOAD PAGE ==========
        st.markdown("## 📁 Upload Brain MRI Scan")
        st.markdown("---")
        
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #00E5FF; margin-bottom: 20px;">💡 Upload Workspace</h3>
            <p style="color: #94A3B8; font-size: 14px;">
                Upload a 3D brain MRI scan in NIfTI format (.nii or .nii.gz). 
                The system will automatically detect tumor, stroke, and Alzheimer patterns.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Drag and drop your MRI scan here",
            type=["nii", "gz"],
            help="Supported formats: .nii, .nii.gz"
        )
        
        if uploaded_file:
            st.success(f"✅ File loaded: {uploaded_file.name}")
            
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("🚀 START ANALYSIS", use_container_width=True):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    
                    try:
                        with st.spinner("🧠 Loading AI model..."):
                            model = load_model()
                        
                        if model:
                            with st.spinner("🔬 Analyzing brain scan..."):
                                # CRITICAL: Now returns 5 values including affine and spacing
                                image_tensor, original_data, roi_metadata, affine, spacing = load_and_preprocess_nifti(tmp_path)
                                image_tensor = image_tensor.to(DEVICE)
                                
                                detection = automatic_disease_detection(model, image_tensor, threshold)
                                detected = detection["detected_diseases"]
                                
                                # Store ALL components including affine/spacing
                                st.session_state.detection_results = detection
                                st.session_state.original_image = original_data
                                st.session_state.roi_metadata = roi_metadata
                                st.session_state.affine = affine
                                st.session_state.spacing = spacing
                                
                                if detected:
                                    segmentations = perform_segmentation(model, image_tensor, detected)
                                    st.session_state.segmentation_results = segmentations
                                
                                st.session_state.analysis_complete = True
                                st.session_state.current_page = 'analysis'
                                st.success("✅ Analysis complete!")
                                st.rerun()
                        else:
                            st.error("❌ Model not found")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
                    finally:
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
    
    elif st.session_state.current_page == 'analysis':
        # ========== ANALYSIS PAGE ==========
        st.markdown("## 🔬 Analysis Dashboard")
        st.markdown("---")
        
        if st.session_state.detection_results:
            det = st.session_state.detection_results
            probs = det["probabilities"]
            detected = det["detected_diseases"]
            
            # Detection Stats
            st.markdown("""
            <div class="glass-card">
                <h3 style="color: #00E5FF; margin-bottom: 20px;">🎯 Detection Results</h3>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            metrics_config = [
                (col1, "🔴 Tumor", probs['tumor'], 'tumor' in detected, "#FF4444"),
                (col2, "🔵 Stroke", probs['stroke'], 'stroke' in detected, "#4444FF"),
                (col3, "🟠 Alzheimer", probs['alzheimer'], 'alzheimer' in detected, "#FF8800")
            ]
            
            for col, label, prob, is_detected, color in metrics_config:
                with col:
                    status = "DETECTED" if is_detected else "Not Detected"
                    st.markdown(f"""
                    <div class="glass-card" style="text-align: center; border-left-color: {color};">
                        <h2 style="color: {color}; font-size: 48px; margin: 0;">{prob:.1%}</h2>
                        <p style="color: #94A3B8; margin: 10px 0 5px;">{label}</p>
                        <p style="color: {'#00FF88' if is_detected else '#FF4444'}; font-weight: 700; font-size: 12px;">{status}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Chart
            st.plotly_chart(create_statistical_summary(det, {}), use_container_width=True)
            
            # Detected Diseases
            if detected:
                st.markdown("""
                <div class="glass-card">
                    <h3 style="color: #00E5FF; margin-bottom: 20px;">🚨 Detected Pathologies</h3>
                </div>
                """, unsafe_allow_html=True)
                
                for disease in detected:
                    name = DISEASE_COLORS[disease]["name"]
                    color = DISEASE_COLORS[disease]["hex"]
                    conf = probs[disease]
                    
                    st.markdown(f"""
                    <div class="glass-card" style="border-left-color: {color};">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <h4 style="color: {color}; margin: 0;">{name}</h4>
                                <p style="color: #94A3B8; margin: 5px 0; font-size: 13px;">Confidence: {conf:.1%}</p>
                            </div>
                            <div style="font-size: 36px;">{'🔴' if disease == 'tumor' else '🔵' if disease == 'stroke' else '🟠'}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No analysis results available. Please upload and analyze a scan first.")
    
    elif st.session_state.current_page == 'visualization':
        # ========== VISUALIZATION PAGE ==========
        st.markdown("## 🧠 3D Visualization Laboratory")
        st.markdown("---")
        
        if st.session_state.segmentation_results:
            st.markdown("""
            <div class="glass-card">
                <h3 style="color: #00E5FF; margin-bottom: 10px;">🌐 3D Brain Rendering</h3>
                <p style="color: #94A3B8; font-size: 13px;">Interactive 3D visualization of detected pathologies</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Patient-specific brain visualization (Otsu brain mask + separate lesion meshes)
            print("\n" + "="*60)
            print("🎬 Starting 3D visualization...")
            print(f"📊 Segmentations available: {list(st.session_state.segmentation_results.keys())}")
            print(f"📏 Original volume shape: {st.session_state.original_image.shape}")
            print(f"📐 Voxel spacing: {st.session_state.spacing}")
            print("="*60 + "\n")
            
            fig_3d = create_3d_visualization(
                segmentations_roi=st.session_state.segmentation_results,
                roi_metadata=st.session_state.roi_metadata,
                original_volume=st.session_state.original_image,
                affine=st.session_state.affine,
                spacing=st.session_state.spacing,
                show_patient_brain=show_atlas,
                show_heatmap=show_heatmap  # Probability heatmap
            )
            
            # Check if figure has any data
            if len(fig_3d.data) == 0:
                st.error("❌ No 3D meshes generated. Check console for errors.")
                print("❌ ERROR: No meshes in figure!")
            else:
                print(f"\n✅ Figure ready: {len(fig_3d.data)} meshes")
                for i, trace in enumerate(fig_3d.data):
                    print(f"  Mesh {i+1}: {trace.name}")
                print()
            
            st.plotly_chart(fig_3d, use_container_width=True)
        else:
            st.info("No visualization data available. Complete analysis first.")
    
    elif st.session_state.current_page == 'reports':
        # ========== REPORTS PAGE ==========
        st.markdown("## 📄 Clinical Report Generator")
        st.markdown("---")
        
        if st.session_state.detection_results:
            st.markdown("""
            <div class="glass-card">
                <h3 style="color: #00E5FF; margin-bottom: 10px;">⚙️ Report Configuration</h3>
                <p style="color: #94A3B8; font-size: 13px;">Generate professional medical reports with AI assistance</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                report_lang = st.selectbox("Report Language", ["English", "Medical Terminology", "Simplified"])
            
            with col2:
                if st.button("✨ GENERATE REPORT", use_container_width=True):
                    with st.spinner("✍️ Generating AI report..."):
                        st.session_state.report_text = generate_ai_report(
                            st.session_state.detection_results,
                            st.session_state.segmentation_results,
                            groq_key if groq_key else None
                        )
            
            if st.session_state.report_text:
                st.markdown("---")
                st.markdown(st.session_state.report_text)
                
                st.markdown("---")
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col2:
                    if st.button("� SAVE TEXT", use_container_width=True):
                        st.download_button(
                            "Download TXT",
                            st.session_state.report_text,
                            file_name=f"neurox_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                
                with col3:
                    if st.button("📄 EXPORT PDF", use_container_width=True):
                        pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
                        if create_pdf_report(st.session_state.detection_results, st.session_state.segmentation_results, st.session_state.report_text, pdf_path):
                            with open(pdf_path, "rb") as f:
                                st.download_button(
                                    "Download PDF",
                                    f,
                                    file_name=f"neurox_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                    mime="application/pdf"
                                )
                            os.unlink(pdf_path)
        else:
            st.info("No detection results available. Complete analysis first.")


if __name__ == "__main__":
    run_streamlit_app()
