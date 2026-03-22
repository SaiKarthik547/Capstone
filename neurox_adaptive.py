

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
from nibabel.processing import resample_from_to
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

# HD-BET availability will be checked once at app startup (in session state)
HDBET_AVAILABLE = None  # Will be set in run_streamlit_app()

print("=" * 60)
print("🔍 HD-BET availability will be checked at app startup")
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
PRESENCE_THRESHOLD = 0.45
# Dynamic Path Configuration
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "checkpoints" / "neurox_model.pth"
ASSET_DIR = BASE_DIR / "assets" / "brain"

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
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.2):
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
        self.bottleneck = TransformerBottleneck3D(128, 4, 8, 256, 0.2)
    
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
        self.dropout = nn.Dropout(0.2)
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
        
        main = self.output_head(d1)
        return main


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention block."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y


class ResBlock3D(nn.Module):
    """Residual Block with InstanceNorm."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
        )
        self.skip = nn.Conv3d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.skip(x))


class AlzheimerEncoder(nn.Module):
    """AlzheimerEncoder v2: Deep Residual + SE Attention.
    
    Structure:
        1 -> 32 (ResBlock) -> Pool
        32 -> 64 (ResBlock) -> Pool
        64 -> 128 (ResBlock) -> Pool
        128 -> 256 (ResBlock) -> SE Attention
        
    Global Modeling: 12x12x12 features with channel attention.
    Classifier: 512 in (Avg+Max concat) -> 256 -> Dropout(0.2) -> 1
    """
    def __init__(self):
        super().__init__()
        self.block1 = ResBlock3D(1, 32)
        self.pool1 = nn.MaxPool3d(2)

        self.block2 = ResBlock3D(32, 64)
        self.pool2 = nn.MaxPool3d(2)

        self.block3 = ResBlock3D(64, 128)
        self.pool3 = nn.MaxPool3d(2)

        self.block4 = ResBlock3D(128, 256)
        self.se = SEBlock(256)

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.norm = nn.LayerNorm(512)

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = self.block4(x)
        x = self.se(x)

        avg = self.avg_pool(x).flatten(1)
        mx  = self.max_pool(x).flatten(1)

        feat = torch.cat([avg, mx], dim=1)
        feat = self.norm(feat)

        return self.classifier(feat)


class NeuroXMultiDisease(nn.Module):
    """Ferrari multitask model with selective forward.

    Alzheimer uses a dedicated AlzheimerEncoder that receives raw MRI directly.
    Zero shared features with segmentation — eliminates representation conflict
    at the encoder level. SharedEncoder is used only for Tumor / Stroke.
    """
    def __init__(self):
        super().__init__()
        self.encoder = SharedEncoder(in_channels=1)
        # Tumor & Stroke presence: transformer bottleneck → PresenceHead
        self.presence_heads = nn.ModuleDict({
            "tumor":  PresenceHead(128),
            "stroke": PresenceHead(128),
            # "alzheimer" removed — uses dedicated AlzheimerEncoder
        })
        self.seg_decoders = nn.ModuleDict({
            "tumor":  SegmentationDecoder(3, "tumor"),   # [ET, NCR, ED]
            "stroke": SegmentationDecoder(1, "stroke")
        })
        # === Alzheimer Dedicated Encoder ===
        # Raw MRI → independent 3D CNN → dual pool → MLP → AD logit
        self.alz_encoder = AlzheimerEncoder()

    def forward(self, x, active_presence=None, active_seg=None):
        features = self.encoder(x)   # enc1, enc2, enc3, bottleneck
        presence = {}
        if active_presence:
            for key in active_presence:
                if key == "alzheimer":
                    # AlzheimerEncoder receives raw MRI directly
                    presence["alzheimer"] = self.alz_encoder(x)
                elif key in self.presence_heads:
                    # Tumor / Stroke: bottleneck → PresenceHead
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
    
    Supports two checkpoint formats:
      - New: dict with 'model_state' + 'metrics' keys (post metrics-history update)
      - Legacy: plain state_dict (old checkpoints load gracefully via fallback)
    """
    model = NeuroXMultiDisease().to(DEVICE)
    if os.path.exists(model_path):
        try:
            # weights_only omitted — dict checkpoints contain non-tensor objects (metrics lists)
            checkpoint = torch.load(model_path, map_location=DEVICE)
            
            # Detect checkpoint format and extract state_dict + metrics
            if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                # New format: {"model_state": ..., "metrics": ...}
                state_dict = checkpoint["model_state"]
                st.session_state.training_metrics = checkpoint.get("metrics", {})
                n_epochs = len(st.session_state.training_metrics.get("epoch", []))
                print(f"✅ New-format checkpoint. Metrics history: {n_epochs} epochs")
            else:
                # Legacy format: plain state_dict — no metrics available
                state_dict = checkpoint
                st.session_state.training_metrics = {}
                print("⚠️ Legacy checkpoint (no metrics). Training dashboard unavailable.")
            
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            # Validate architecture
            assert "alzheimer" not in model.seg_decoders, "Alzheimer must not have segmentation decoder"
            assert "alzheimer" not in model.presence_heads, "Alzheimer must use dedicated AlzheimerEncoder"
            assert hasattr(model, "alz_encoder"), "AlzheimerEncoder (model.alz_encoder) missing from model"
            assert isinstance(model.alz_encoder, AlzheimerEncoder), "model.alz_encoder must be AlzheimerEncoder"
            
            model.eval()
            
            # Warn about mismatched keys (expected when architecture changed from global branch to dedicated encoder)
            if missing_keys or unexpected_keys:
                warning_msg = f"⚠️ Checkpoint partial match: {len(missing_keys)} missing, {len(unexpected_keys)} unexpected\n"
                warning_msg += "Expected if loading old checkpoints (alz_pool/alz_norm/alz_classifier → alz_encoder)."
                st.info(warning_msg)
                print(warning_msg)
            
            print(f"✅ Model loaded: {model_path}")
            print(f"✅ Architecture validated: Alzheimer dedicated encoder, no segmentation decoder")
            return model
        except Exception as e:
            st.error(f"Model loading failed: {e}")
            traceback.print_exc()
            return None
    else:
        st.warning(f"⚠️ Model file not found: {model_path}")
        return None


def load_and_preprocess_nifti(file_path: str) -> Tuple[torch.Tensor, np.ndarray, Dict, np.ndarray, Tuple]:
    """Load and preprocess NIfTI file identifying with training baseline.
    
    Pipeline:
    1. Enforce canonical RAS+ orientation (nib.as_closest_canonical)
    2. Z-score normalize
    3. Direct trilinear resize to 96³
    4. Track roi_affine for affine-aware inverse resampling
    """
    img = nib.load(file_path)
    # FIX 1.1 — Enforce canonical RAS+ orientation before ANY processing.
    # Without this, scanner-specific axis permutations silently corrupt all
    # numpy indexing downstream (z/y/x swap → mesh misalignment).
    img = nib.as_closest_canonical(img)
    
    original_data = img.get_fdata().astype(np.float32)
    original_shape = original_data.shape
    
    # Extract affine AFTER reorientation (affine changes with canonical)
    affine = img.affine
    spacing_raw = np.sqrt(np.sum(affine[:3, :3]**2, axis=0))
    # ISSUE 6 FIX: Explicit spacing validation with anisotropy check
    if np.any(spacing_raw <= 0) or np.any(np.isnan(spacing_raw)):
        print(f'   WARNING: Invalid spacing {spacing_raw} -- using isotropic 1mm fallback')
        spacing = (1.0, 1.0, 1.0)
    else:
        max_ratio = spacing_raw.max() / (spacing_raw.min() + 1e-8)
        if max_ratio > 10.0:
            print(f'   WARNING: Extreme anisotropy ({max_ratio:.1f}x): {spacing_raw} mm')
        else:
            print(f'   Voxel spacing: {spacing_raw[0]:.2f} x {spacing_raw[1]:.2f} x {spacing_raw[2]:.2f} mm')
        spacing = tuple(float(s) for s in spacing_raw)
    
    # Preprocess
    data = original_data.copy()
    if data.ndim == 4:
        data = data[..., 0] if data.shape[-1] <= 2 else data[..., :2].mean(axis=-1)
        
    mean, std = data.mean(), data.std() + 1e-8
    data_normalized = (data - mean) / std
    
    volume_tensor = torch.from_numpy(data_normalized).unsqueeze(0).unsqueeze(0)
    
    # Direct resize to 96³ (matches training preprocessing)
    roi_tensor = F.interpolate(
        volume_tensor,
        size=ROI_SIZE,
        mode='trilinear',
        align_corners=False
    )
    
    # FIX 1.2/1.4 — Compute ROI's own affine for affine-aware inverse resampling.
    # Since preprocessing is a center-preserving full-volume resize (no crop),
    # the ROI affine shares the same origin but has scaled voxel sizes.
    orig_shape_3d = tuple(original_shape) if original_data.ndim == 3 else tuple(original_shape[:3])
    scale = np.array(orig_shape_3d, dtype=np.float64) / np.array(ROI_SIZE, dtype=np.float64)
    roi_affine = affine.copy()
    roi_affine[:3, :3] = affine[:3, :3] * scale[np.newaxis, :]  # Scale voxel size, keep origin
    
    roi_metadata = {
        "original_shape":  orig_shape_3d,
        "interpolation_mode": "trilinear",
        "roi_affine":      roi_affine,    # ROI (96³) physical coordinate system
        "original_affine": affine,        # Original image coordinate system (for resample_from_to)
    }
    
    return roi_tensor, original_data, roi_metadata, affine, spacing


def compute_lesion_metrics(segmentation: np.ndarray, affine: np.ndarray) -> Optional[Dict]:
    """Clinical-style spatial and volumetric quantification using affine.
    
    CRITICAL: Corrects for axis swap between numpy (z,y,x) and nibabel (x,y,z).
    
    Args:
        segmentation: Binary segmentation mask (3D uint8)
        affine: 4x4 affine matrix for world coordinate mapping
        
    Returns:
        Structured dictionary or None if no lesion detected.
    """
    import nibabel as nib
    
    # Ensure 3D
    if segmentation.ndim != 3:
        if segmentation.ndim == 4:
            segmentation = segmentation.max(axis=0)
        else:
            return None

    # Get voxel coordinates of non-zero elements
    coords_voxel = np.argwhere(segmentation > 0)
    
    if len(coords_voxel) == 0:
        return None

    # 1. Volume calculation
    # Determinant of 3x3 rotational part gives voxel volume
    voxel_vol = np.abs(np.linalg.det(affine[:3, :3]))
    volume_mm3 = float(len(coords_voxel) * voxel_vol)

    # 2. Centroid in World Coordinates
    # Important: argwhere returns (z, y, x), but apply_affine expects (x, y, z)
    centroid_voxel_zyx = coords_voxel.mean(axis=0)
    centroid_voxel_xyz = centroid_voxel_zyx[[2, 1, 0]]
    centroid_world = nib.affines.apply_affine(affine, centroid_voxel_xyz)

    # 3. Bounding Box in World Coordinates
    min_voxel_zyx = coords_voxel.min(axis=0)
    max_voxel_zyx = coords_voxel.max(axis=0)
    
    # Swap to XYZ before applying affine
    min_world = nib.affines.apply_affine(affine, min_voxel_zyx[[2, 1, 0]])
    max_world = nib.affines.apply_affine(affine, max_voxel_zyx[[2, 1, 0]])

    return {
        "volume_mm3": volume_mm3,
        "centroid_mm": centroid_world.tolist(),
        "bbox_min_mm": min_world.tolist(),
        "bbox_max_mm": max_world.tolist(),
        "voxel_count": int(len(coords_voxel))
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

    Alzheimer uses the dedicated AlzheimerEncoder (raw MRI → 3D CNN → dual pool → MLP)
    instead of the shared encoder path — MC-Dropout uncertainty via Dropout(0.25) in
    alz_encoder.classifier.
    """
    if model is None:
        return {"detected_diseases": [], "probabilities": {}, "uncertainties": {}}
    
    probabilities = {}
    uncertainties = {}
    presence_logits = {}
    
    with torch.no_grad():
        features   = model.encoder(image_tensor.to(DEVICE))
        bottleneck = features["bottleneck"]
        # enc3 no longer needed — Alzheimer uses dedicated AlzheimerEncoder
        
        # ── Tumor & Stroke via PresenceHead (bottleneck) ──────────────────
        for disease in ["tumor", "stroke"]:
            head = model.presence_heads[disease]
            logit = head(bottleneck)
            
            if use_uncertainty:
                mean_prob, uncertainty = head.uncertainty_forward(bottleneck, n_samples=10)
                mean_prob = float(np.clip(mean_prob, 1e-6, 1.0 - 1e-6))
                probabilities[disease] = mean_prob
                uncertainties[disease] = uncertainty
                presence_logits[disease] = np.log(mean_prob / (1 - mean_prob))
            else:
                presence_logits[disease] = float(logit.cpu().item())
                probabilities[disease] = torch.sigmoid(logit).cpu().item()
                uncertainties[disease] = 0.0
        
    # ── Alzheimer via dedicated AlzheimerEncoder (raw MRI) ───────────────
    img_dev = image_tensor.to(DEVICE)
    
    if use_uncertainty:
        # MC-Dropout: enable Dropout(0.25) inside alz_encoder.classifier
        model.alz_encoder.classifier.train()
        alz_samples = []
        with torch.no_grad():
            for _ in range(10):
                logit = model.alz_encoder(img_dev)
                alz_samples.append(torch.sigmoid(logit).cpu().item())
        model.alz_encoder.classifier.eval()

        mean_prob    = float(np.mean(alz_samples))
        mean_prob    = float(np.clip(mean_prob, 1e-6, 1.0 - 1e-6))
        uncertainty  = float(np.std(alz_samples))
        probabilities["alzheimer"]   = mean_prob
        uncertainties["alzheimer"]   = uncertainty
        presence_logits["alzheimer"] = float(np.log(mean_prob / (1 - mean_prob)))
    else:
        with torch.no_grad():
            logit = model.alz_encoder(img_dev)
        presence_logits["alzheimer"] = float(logit.cpu().item())
        probabilities["alzheimer"]   = torch.sigmoid(logit).cpu().item()
        uncertainties["alzheimer"]   = 0.0
    
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
    """Segment detected diseases based on training baseline (raw thresholding)."""
    if model is None:
        return {}
    
    # Filter to only segmentable diseases
    seg_diseases = [d for d in diseases if d in ["tumor", "stroke"]]
    
    if not seg_diseases:
        return {}
    
    results = {}
    with torch.no_grad():
        output = model(image_tensor.to(DEVICE), active_presence=None, active_seg=seg_diseases)
        
        for disease in seg_diseases:
            seg_logits = output["segmentations"][disease]
            seg_probs = torch.sigmoid(seg_logits).cpu().numpy()[0]
            
            # Post-process: Simple thresholding (No morphology as per training)
            seg_binary = (seg_probs > 0.5).astype(np.uint8)
            
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
    """
    FIX 1.3 — Affine-aware inverse resampling: ROI (96³) → original image space.

    Uses nibabel resample_from_to so that voxel positions are mapped through
    physical coordinates (mm), not pixel-count ratios. This correctly handles
    anisotropic volumes (e.g. 240×240×155 → 96³) where naive F.interpolate
    stretches each axis at a different ratio causing spatial drift.

    For 4D inputs (tumor C×D×H×W) each channel is resampled independently.
    """
    from nibabel.processing import resample_from_to

    target_shape = roi_metadata["original_shape"]
    roi_affine   = roi_metadata.get("roi_affine")
    orig_affine  = roi_metadata.get("original_affine")

    # Legacy fallback: old metadata without affines (e.g. cached session state from before fix)
    if roi_affine is None or orig_affine is None:
        print("⚠️  roi_affine not found in metadata — using legacy F.interpolate fallback")
        seg_tensor = torch.from_numpy(seg_roi).float()
        if seg_tensor.ndim == 3:
            seg_tensor = seg_tensor.unsqueeze(0).unsqueeze(0)
        else:
            seg_tensor = seg_tensor.unsqueeze(0)
        seg_resampled = F.interpolate(seg_tensor, size=target_shape, mode='nearest').squeeze(0)
        if seg_roi.ndim == 3:
            seg_resampled = seg_resampled.squeeze(0)
        return seg_resampled.cpu().numpy().astype(np.uint8)

    # 4D: process each channel independently (tumor: ET / NCR / ED)
    if seg_roi.ndim == 4:
        channels = []
        for c in range(seg_roi.shape[0]):
            ch_nifti     = nib.Nifti1Image(seg_roi[c].astype(np.float32), roi_affine)
            ch_resampled = resample_from_to(ch_nifti, (target_shape, orig_affine), order=0)
            channels.append(ch_resampled.get_fdata().astype(np.uint8))
        return np.stack(channels, axis=0)

    # 3D: single channel (stroke, or collapsed tumor Whole Tumor mask)
    seg_nifti     = nib.Nifti1Image(seg_roi.astype(np.float32), roi_affine)
    seg_resampled = resample_from_to(seg_nifti, (target_shape, orig_affine), order=0)
    return seg_resampled.get_fdata().astype(np.uint8)


def validate_alignment(segmentation_mask: np.ndarray, brain_mask: np.ndarray) -> None:
    """Validate spatial alignment between segmentation and brain mask."""
    if segmentation_mask.sum() == 0:
        print("⚠️ Segmentation empty — skipping validation.")
        return

    if brain_mask is None:
        return

    overlap = (segmentation_mask & brain_mask).sum()
    ratio = overlap / (segmentation_mask.sum() + 1e-6)

    print(f"   🔍 Alignment Check: {ratio * 100:.2f}% of lesion inside brain")

    if ratio < 0.80:
        print("   ⚠️ WARNING: Possible spatial misalignment detected (< 80% overlap)")
    else:
        print("   ✅ Segmentation spatially consistent")


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
    
    # FIX 4.2 — Unified Gaussian sigma=1.0 (matches lesion smoothing)
    from scipy.ndimage import gaussian_filter
    brain_smooth = gaussian_filter(brain_mask.astype(float), sigma=1.0)
    
    # Marching cubes at level=0.5 (standard for binary masks)
    # FIX 2.3 — NO spacing argument. The affine matrix already encodes
    # physical voxel size in mm via its column vectors. Passing spacing=spacing
    # here would pre-scale the vertices before the affine is applied,
    # causing double-scaling relative to the lesion mesh (which has no spacing).
    try:
        from skimage import measure
        verts, faces, normals, _ = measure.marching_cubes(
            brain_smooth,
            level=0.5
        )
    except (ValueError, RuntimeError) as e:
        raise RuntimeError(f"Brain surface generation failed: {e}")
    
    # Apply affine transform if provided (convert to world coordinates)
    if affine is not None:
        # Marching cubes returns vertices in (z, y, x) order — swap to (x, y, z) for affine
        verts_xyz = verts[:, [2, 1, 0]]
        # FIX 4.4 — use nib.affines.apply_affine (cleaner, equivalent, avoids manual homogeneous coords)
        verts = nib.affines.apply_affine(affine, verts_xyz)
    
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
            
            # Load results - HD-BET outputs brain-extracted volume
            brain_path = output_path  # output.nii.gz
            
            print(f"📂 Loading HD-BET output...")
            print(f"   Brain volume: {brain_path}")
            
            if not os.path.exists(brain_path):
                print(f"❌ Brain volume not found: {brain_path}")
                print(f"   Files in temp dir: {os.listdir(tmpdir)}")
                return None, None
            
            # ISSUE 2 FIX: Enforce canonical orientation on HD-BET output.
            # HD-BET may internally reorient the volume before writing output.
            # Without this the brain mask affine can silently diverge from the
            # input affine, causing the brain surface and lesion meshes to split.
            brain_img = nib.load(brain_path)
            brain_img = nib.as_closest_canonical(brain_img)   # enforce RAS+
            brain_volume = brain_img.get_fdata()
            brain_affine = brain_img.affine

            input_shape = volume.shape
            affine_ok  = np.allclose(brain_affine, affine, atol=1e-3)
            shape_ok   = (brain_volume.shape == input_shape)
            if not affine_ok or not shape_ok:
                print("WARNING: HD-BET geometry mismatch after canonical enforcement!")
                print("  Resampling back to master grid...")
                
                # Master geometry defined by (input_shape, affine)
                # We need a reference NIfTI image for resampling
                target_img = nib.Nifti1Image(np.zeros(input_shape), affine)
                
                # Resample (Nearest neighbor order=0 for masks/labels)
                brain_resampled_img = resample_from_to(
                    brain_img,
                    target_img,
                    order=0  # Nearest neighbor
                )
                
                brain_volume = brain_resampled_img.get_fdata()
                print(f"✅ Resampling complete. Brain mask now aligned to Master Grid.")
            
            # Generate binary mask from brain volume
            # HD-BET sets non-brain voxels to 0, brain voxels to original intensity
            brain_mask = (brain_volume > 0).astype(bool)
            
            print(f"✅ Brain volume loaded successfully")
            print(f"   Generating binary mask from brain volume...")
            
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
    show_heatmap: bool = False,
    lesion_metrics: Optional[Dict] = None
) -> go.Figure:
    """Medical-Grade Patient-Specific Brain Visualization.
    
    ANATOMICALLY ACCURATE SURFACE RENDERING:
    1. Generate brain mask using Otsu thresholding
    2. Crop to brain bounding box
    3. Brain surface from mask (marching cubes at level=0.5)
    4. Lesion surfaces separately (never merged with brain)
    5. Apply affine transforms (world coordinates)
    
    Args:
        segmentations_roi: Dict of (probs, binary) in ROI space (96³)
        roi_metadata: Coordinate mapping metadata
        original_volume: Full-resolution patient MRI
        affine: NIfTI affine matrix
        spacing: Voxel spacing in mm
        show_patient_brain: Whether to render patient brain surface
        clinical_decision: Optional clinical analysis with primary disease
        show_heatmap: Whether to show probability heatmap
        lesion_metrics: Pre-computed world-coordinate metrics
    
    Returns:
        Plotly 3D figure with world-coordinate meshes and annotations
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
    
    # CRITICAL: Compute affine for the CROPPED region to support world coordinates
    # Translation matrix for cropping offset (in voxel space)
    # nibabel (and our corrected generation) expects XYZ order
    z_start, y_start, x_start = brain_bbox[0].start, brain_bbox[1].start, brain_bbox[2].start
    T = np.eye(4)
    T[:3, 3] = [x_start, y_start, z_start] 
    cropped_affine = affine @ T
    print("🌍 Scene shifted to WORLD COORDINATES (mm)")
    print("=" * 60 + "\n")
    
    # LAYER 1: Patient-Specific Brain Surface (from mask, not MRI)
    if show_patient_brain and brain_mask_cropped is not None:
        try:
            with st.spinner("Generating patient brain surface..."):
                print("🧠 Generating brain surface mesh...")
                brain_verts, brain_faces = generate_patient_brain_surface(
                    brain_mask=brain_mask_cropped,
                    affine=cropped_affine,
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
        
        # Use same threshold as training
        VIS_THRESHOLD = 0.5
        
        print(f"\n📊 {name} Probability Distribution in ROI:")
        print(f"   Min: {probs_roi.min():.4f}")
        print(f"   Max: {probs_roi.max():.4f}")
        print(f"   ROI sum:", (probs_roi > 0.5).sum())
        print(f"   Voxels > 0.5: {(probs_roi > 0.5).sum():,} ({(probs_roi > 0.5).sum()/probs_roi.size*100:.1f}%)")
        
        if disease == "tumor" and probs_roi.ndim == 4:
            # FIX 5.2 — Correct BraTS channel order from SegmentationDecoder:
            # Channel 0 = ET  (Enhancing Tumor, BraTS label 4)
            # Channel 1 = NCR (Necrotic Core,   BraTS label 1)
            # Channel 2 = ED  (Edema,            BraTS label 2)
            prob_et  = probs_roi[0]   # Enhancing Tumor
            prob_ncr = probs_roi[1]   # Necrotic Core
            prob_ed  = probs_roi[2]   # Edema
            # Whole Tumor (WT) = union of all sub-regions
            prob_wt = np.maximum.reduce([prob_et, prob_ncr, prob_ed])
            
            selected_source = "Whole Tumor (ET + NCR + ED)"
            print(f"   Selected: {selected_source} with threshold {VIS_THRESHOLD}")
            st.caption(f"Visualizing: **{selected_source}** (Threshold: {VIS_THRESHOLD:.2f})")
            
            binary_strict = (prob_wt > VIS_THRESHOLD).astype(np.uint8)
            
        elif probs_roi.ndim == 4:  # Multi-channel logic (generic fallback)
            binary_strict = (probs_roi.max(axis=0) > VIS_THRESHOLD).astype(np.uint8)
        else:
            # Single channel (Stroke)
            binary_strict = (probs_roi > VIS_THRESHOLD).astype(np.uint8)
        
        print(f"   After threshold {VIS_THRESHOLD}: {binary_strict.sum():,} voxels in ROI")
        
        # CRITICAL: Map from ROI space to original patient space
        seg_original = map_segmentation_to_original_space(binary_strict, roi_metadata)
        print(f"   Original space: {seg_original.shape}, {seg_original.sum():,} voxels")

        # ISSUE 3 FIX: Guard against lesion vanishing after resampling.
        # Tiny ROI lesions can fall between grid points during nearest-neighbor
        # upsampling from 96^3 back to the larger original space.
        if seg_original.sum() == 0:
            st.warning(
                f"Segmentation for {name} vanished after resampling. "
                f"The ROI lesion ({int(binary_strict.sum())} voxels) was too small "
                f"to survive upsampling from 96^3 to {roi_metadata['original_shape']}."
            )
            print(f"   Lesion disappeared after resampling -- skipping {name}")
            continue

        # FIX 3.1 — Assert shapes match BEFORE cropping. If this fires, resampling is broken.
        assert seg_original.shape == original_volume.shape[:3], (
            f"Segmentation shape {seg_original.shape} does not match "
            f"original volume shape {original_volume.shape[:3]}. "
            "resample_from_to fix must have failed — check roi_affine in roi_metadata."
        )
        
        # -------------------------------------------------
        # ALIGNMENT VALIDATION (FULL MASTER GRID)
        # -------------------------------------------------
        # Strictly validate against FULL brain mask BEFORE any cropping
        validate_alignment(seg_original, brain_mask)
        
        # Crop to brain bounding box ONLY AFTER validation
        if brain_bbox is not None:
            seg_original = seg_original[brain_bbox]
            print(f"   After bbox crop: {seg_original.shape}, {seg_original.sum():,} voxels")
        
        # ═══════════════════════════════════════════════════════════════
        # ANATOMICAL CONSTRAINT PIPELINE (Post-Resample)
        # ROOT CAUSE FIX: 933 ROI voxels expand to 35k+ in original space.
        # Resampling amplifies small errors into floating clusters outside brain.
        # We enforce hard anatomical constraints before ANY mesh generation.
        # ═══════════════════════════════════════════════════════════════
        from scipy.ndimage import label as cc_label

        # FIX 1 — HARD CLIP: Force lesion inside brain mask (non-negotiable).
        # Any voxel outside the brain mask is anatomically impossible.
        before_clip = int(seg_original.sum())
        seg_original = seg_original.astype(bool) & brain_mask_cropped.astype(bool)
        after_clip = int(seg_original.sum())
        print(f"   Brain mask hard clip: {before_clip:,} → {after_clip:,} voxels "
              f"({before_clip - after_clip:,} outside-brain voxels removed)")

        # FIX 2 — LARGEST CONNECTED COMPONENT: Eliminate floating clusters.
        # Resampling creates disconnected specks that produce phantom meshes.
        labeled, num_components = cc_label(seg_original)
        if num_components > 1:
            sizes = [(labeled == i).sum() for i in range(1, num_components + 1)]
            largest_label = int(np.argmax(sizes)) + 1
            # FIX 3 — MICRO-ARTIFACT REMOVAL: Drop components < 100 voxels.
            MIN_COMPONENT_SIZE = 100
            seg_original = np.zeros_like(seg_original, dtype=bool)
            for i in range(1, num_components + 1):
                comp_size = sizes[i - 1]
                if comp_size >= MIN_COMPONENT_SIZE:
                    seg_original |= (labeled == i)
            kept = int(seg_original.sum())
            print(f"   Connected components: {num_components} found, "
                  f"kept voxels >={MIN_COMPONENT_SIZE}: {kept:,}")
        else:
            print(f"   Connected components: 1 (no fragmentation)")

        seg_original = seg_original.astype(np.uint8)

        # FIX 4 — STRICT VALIDATION: Require ≥98% of lesion inside brain.
        # Replace old soft 40% check with a hard clinical-grade threshold.
        if brain_mask_cropped is not None and seg_original.sum() > 0:
            inside = int((seg_original.astype(bool) & brain_mask_cropped.astype(bool)).sum())
            total  = int(seg_original.sum())
            inside_ratio = inside / (total + 1e-8)
            print(f"   Strict alignment: {inside_ratio:.2%} inside brain ({inside:,}/{total:,})")
            if inside_ratio < 0.98:
                st.warning(
                    f"⚠️ **{name}**: Poor alignment ({inside_ratio:.1%} inside brain). "
                    f"Applying additional brain-mask clamp..."
                )
                # Apply a second clamp pass to push ratio to 100%
                seg_original = (seg_original.astype(bool) & brain_mask_cropped.astype(bool)).astype(np.uint8)
                print(f"   After second clamp: {seg_original.sum():,} voxels")

        seg_clean = seg_original
        cleaned_voxels = int(seg_clean.sum())
        print(f"   Final lesion voxels for mesh: {cleaned_voxels:,}")

        # FIX 4.3 — Minimum volume check (after all cleaning)
        MIN_VOLUME_VOXELS = 27
        if cleaned_voxels < MIN_VOLUME_VOXELS:
            st.warning(f"⚠️ **{name}**: Lesion too small after cleaning ({cleaned_voxels} voxels).")
            print(f"   ⚠️ Skipping: only {cleaned_voxels} voxels (minimum {MIN_VOLUME_VOXELS})")
            continue
        
        # FIX 4.2 — Unified sigma=1.0 to match brain surface smoothing
        sigma = 1.0
        
        try:
            seg_smooth = gaussian_filter(seg_clean.astype(float), sigma=sigma)
            
            # GOLD-STANDARD: Marching cubes in voxel space
            verts, faces, normals, _ = measure.marching_cubes(
                seg_smooth,
                level=0.5
            )
            
            # FIX 4.4 — ZYX → XYZ swap then use nib.affines.apply_affine (cleaner than manual homogeneous)
            verts_xyz = verts[:, [2, 1, 0]]
            verts = nib.affines.apply_affine(cropped_affine, verts_xyz)
            
            # FIX 8.1 — Mesh decimation: cap at 80k verts to prevent browser crash
            MAX_VERTS = 80_000
            if len(verts) > MAX_VERTS:
                step = max(2, len(verts) // MAX_VERTS)
                keep_mask = np.zeros(len(verts), dtype=bool)
                keep_mask[::step] = True
                old_to_new = np.full(len(verts), -1, dtype=np.int64)
                old_to_new[keep_mask] = np.arange(keep_mask.sum())
                verts = verts[keep_mask]
                # Keep only faces where ALL 3 vertices survived decimation
                face_mask = np.all(old_to_new[faces] >= 0, axis=1)
                faces = old_to_new[faces[face_mask]]
                print(f"   ⚡ Decimated lesion mesh: {len(verts):,} verts retained")
            
            # Show vertex range for debugging
            print(f"   Vertex range: X=[{verts[:,0].min():.1f}, {verts[:,0].max():.1f}]")
            print(f"                 Y=[{verts[:,1].min():.1f}, {verts[:,1].max():.1f}]")
            print(f"                 Z=[{verts[:,2].min():.1f}, {verts[:,2].max():.1f}]")
            
            # Prepare mesh coloring
            if show_heatmap:
                # Heatmap: sample probability values at vertex locations
                with st.spinner(f"Mapping {name} probabilities for heatmap..."):
                    # 1. Map ROI probabilities back to original space (float)
                    target_shape = roi_metadata["original_shape"]
                    prob_tensor = torch.from_numpy(probs_roi).float()
                    if prob_tensor.ndim == 3:
                        prob_tensor = prob_tensor.unsqueeze(0).unsqueeze(0)
                    else:
                        prob_tensor = prob_tensor.unsqueeze(0)
                        
                    prob_original = F.interpolate(
                        prob_tensor,
                        size=target_shape,
                        mode='trilinear',
                        align_corners=False
                    ).squeeze().cpu().numpy()
                    
                    # 2. Crop probability map to same brain bbox
                    prob_cropped = prob_original[brain_bbox]
                    
                    # 3. Get sampling coordinates using INVERSE AFFINE for 100% sync
                    from scipy.ndimage import map_coordinates
                    inv_affine = np.linalg.inv(cropped_affine)
                    
                    # Verts are currently world coordinates (mm)
                    # apply_affine(inv, world) -> xyz voxel coordinates
                    verts_xyz_recon = nib.affines.apply_affine(inv_affine, verts)
                    
                    # 4. map_coordinates expects (z, y, x) i.e. (d, h, w)
                    verts_zyx_recon = verts_xyz_recon[:, [2, 1, 0]]
                    
                    # Sample probabilities at precise mesh vertex locations
                    vertex_probs = map_coordinates(
                        prob_cropped,
                        verts_zyx_recon.T,
                        order=1,
                        mode='nearest'
                    )
                    
                    print(f"   🔥 Heatmap sampled via inverse affine from world coords")
                    print(f"   Prob range: [{vertex_probs.min():.4f}, {vertex_probs.max():.4f}]")
                
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
            
            # ─── ADD SCENE ANNOTATIONS (WORLD COORDINATES) ───
            if lesion_metrics and disease in lesion_metrics:
                m = lesion_metrics[disease]
                if m:
                    centroid = m["centroid_mm"]
                    vol_ml = m["volume_mm3"] / 1000.0
                    
                    fig.add_trace(go.Scatter3d(
                        x=[centroid[0]], y=[centroid[1]], z=[centroid[2]],
                        mode='markers+text',
                        marker=dict(size=8, color=color, symbol='diamond', 
                                   line=dict(color='white', width=2)),
                        text=[f"<b>{name}</b><br>{vol_ml:.2f} mL<br>({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f})"],
                        textposition="top center",
                        textfont=dict(color='white', size=12),
                        name=f"{name} Center",
                        hoverinfo='text'
                    ))
                    print(f"   📍 Added centroid marker at world: {centroid}")
            
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

Generate an educational radiology report template for research purposes that demonstrates proper medical imaging documentation standards. This report will be used to illustrate how radiologists communicate imaging findings while maintaining appropriate clinical caution and scientific accuracy. The report specifically addresses structural MRI findings, utilizing T1, T2, and FLAIR sequences to evaluate hippocampal morphology.

Task
The assistant should generate a sample radiology-style report that describes imaging patterns and characteristics observed on structural MRI without making definitive diagnostic claims. The report must use appropriate hedging language, emphasize the need for clinical correlation, and include a dedicated limitations section. The output should be structured in three sections: Findings, Impression, and Limitations. Focus specifically on hippocampal volume and morphological characteristics as detected on T1, T2, and FLAIR sequences.

Objective
To create an educational document that demonstrates best practices in medical imaging reporting, including appropriate use of cautious language, acknowledgment of technical limitations, and the importance of multidisciplinary clinical assessment in interpreting structural MRI studies for neurodegenerative pattern recognition.

Knowledge

Alzheimer's disease detection in imaging is based on the presence of specific patterns (such as atrophy distribution and hippocampal morphology), not volumetric measurements alone
Structural MRI with T1, T2, and FLAIR sequences provides complementary information: T1 sequences are optimal for anatomical detail and volumetric assessment, T2 sequences detect signal abnormalities, and FLAIR sequences suppress cerebrospinal fluid to enhance detection of subtle pathology
Educational radiology reports must avoid diagnostic certainty and instead describe "imaging characteristics consistent with" or "patterns suggestive of" potential conditions
All findings must emphasize the requirement for clinical correlation with patient history, cognitive testing, and other diagnostic modalities
Professional medical terminology should be maintained throughout
Technical limitations of structural MRI must be explicitly stated, including inability to detect microscopic pathology and dependence on sequence optimization
The report should not exceed 300 words total
Output Structure:

FINDINGS
[Describe detected patterns on T1, T2, and FLAIR sequences, with specific attention to hippocampal morphology]

IMPRESSION
[Clinical correlation needed statement]

LIMITATIONS
[Technical limitations specific to structural MRI]
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


        /* HIDE SIDEBAR COLLAPSE BUTTON */
        [data-testid="stSidebarCollapseButton"] {
            display: none !important;
        }
        
        /* HIDE SIDEBAR COMPLETELY */
        [data-testid="stSidebar"] {
            display: none !important;
        }
        
        [data-testid="stSidebarNav"] {
            display: none !important;
        }
        
        section[data-testid="stSidebar"] {
            display: none !important;
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
    if 'affine' not in st.session_state:
        st.session_state.affine = None
    if 'spacing' not in st.session_state:
        st.session_state.spacing = None
    if 'roi_metadata' not in st.session_state:
        st.session_state.roi_metadata = {}
    if 'report_text' not in st.session_state:
        st.session_state.report_text = ""
    if 'training_metrics' not in st.session_state:
        st.session_state.training_metrics = {}   # populated by load_model from checkpoint
        
    # GLOBAL SETTINGS STATE (Initialize defaults)
    if 'show_atlas' not in st.session_state:
        st.session_state.show_atlas = True
    if 'show_heatmap' not in st.session_state:
        st.session_state.show_heatmap = False
    if 'confidence_threshold' not in st.session_state:
        st.session_state.confidence_threshold = PRESENCE_THRESHOLD
    
    # HD-BET AVAILABILITY CHECK (RUNS ONLY ONCE AT STARTUP)
    if 'hdbet_available' not in st.session_state:
        global HDBET_AVAILABLE
        print("\n" + "=" * 60)
        print("🔍 Checking HD-BET availability (ONE-TIME CHECK)...")
        
        # Configure HD-BET to use local weights
        local_weights_path = BASE_DIR / "release_v1.5.0" / "fold_all"
        if os.path.exists(local_weights_path):
            os.environ['HDBET_WEIGHTS'] = str(local_weights_path)
            print(f"✅ Local HD-BET weights found: {local_weights_path}")
            print(f"   checkpoint_final.pth will be used for brain extraction")
        
        try:
            result = subprocess.run(
                ["hd-bet", "-h"],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                HDBET_AVAILABLE = True
                st.session_state.hdbet_available = True
                print("✅ HD-BET CLI is available and working")
                print("🎯 HD-BET will be used for medical-grade brain extraction")
                print("   (Using local checkpoint from release_v1.5.0/)")
            else:
                HDBET_AVAILABLE = False
                st.session_state.hdbet_available = False
                print("❌ HD-BET CLI returned error")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            HDBET_AVAILABLE = False
            st.session_state.hdbet_available = False
            print(f"❌ HD-BET CLI not found: {e}")
            print("⚠️  3D brain surface rendering will be DISABLED")
        
        print("=" * 60 + "\n")
    else:
        # Use cached value
        HDBET_AVAILABLE = st.session_state.hdbet_available
    
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
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
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

    with col5:
        if st.button("⚙️ SETTINGS", key="nav_settings", use_container_width=True):
            st.session_state.current_page = 'settings'
            st.rerun()

    with col6:
        if st.button("📈 TRAINING", key="nav_train", use_container_width=True):
            st.session_state.current_page = 'training'
            st.rerun()
    
    # REMOVED SIDEBAR: All controls now in Settings page
    
    
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
            help="Supported formats: .nii.gz"
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
                                
                                # Use session safe threshold
                                thr = st.session_state.confidence_threshold
                                detection = automatic_disease_detection(model, image_tensor, thr)
                                
                                # TEMPORARY: Bypass detection gating for segmentation validation
                                # perform segmentation always for tumor and stroke
                                detected_for_seg = ["tumor", "stroke"]
                                
                                # Store ALL components including affine/spacing
                                st.session_state.detection_results = detection
                                st.session_state.original_image = original_data
                                st.session_state.roi_metadata = roi_metadata
                                st.session_state.affine = affine
                                st.session_state.spacing = spacing
                                
                                segmentations = perform_segmentation(model, image_tensor, detected_for_seg)
                                st.session_state.segmentation_results = segmentations
                                
                                # CRITICAL: Calculate metrics after segmentation and store in session state
                                lesion_metrics = {}
                                for disease, (_, binary_roi) in segmentations.items():
                                    # Map to original space first for correct coordinates
                                    seg_original = map_segmentation_to_original_space(binary_roi, roi_metadata)
                                    lesion_metrics[disease] = compute_lesion_metrics(seg_original, affine)
                                
                                st.session_state.lesion_metrics = lesion_metrics
                                
                                st.session_state.analysis_complete = True
                                st.session_state.current_page = 'analysis'
                                st.success("✅ Analysis complete!")
                                st.rerun()
                        else:
                            st.error("❌ Model not found")
                    except Exception as e:
                        import traceback
                        st.error(f"❌ Analysis failed: {type(e).__name__}: {e}")
                        st.code(traceback.format_exc(), language="python")
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
            
            # Comprehensive Clinical Metrics Section
            st.markdown("---")
            st.markdown("""
            <div class="glass-card">
                <h3 style="color: #00E5FF; margin-bottom: 20px;">📊 Comprehensive Clinical Metrics</h3>
                <p style="color: #94A3B8; font-size: 13px;">Detailed performance metrics for each disease classification</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Calculate comprehensive metrics for each disease
            for disease in ["tumor", "stroke", "alzheimer"]:
                disease_name = DISEASE_COLORS[disease]["name"]
                disease_color = DISEASE_COLORS[disease]["hex"]
                prob = probs[disease]
                is_detected = disease in detected
                
                # Get uncertainty if available
                uncertainty = det.get("uncertainties", {}).get(disease, 0.0)
                
                # Display metrics in expandable section
                with st.expander(f"📈 {disease_name} - Detection Metrics", expanded=is_detected):
                    st.markdown(f"""
                    <div class="glass-card">
                        <p style="color: #94A3B8; font-size: 13px; margin-bottom: 15px;">
                            ⚠️ <b>Note:</b> Comprehensive performance metrics (Sensitivity, Specificity, PPV, NPV) 
                            require validation against ground truth labels. The values shown below are the model's 
                            confidence scores from inference.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Model Confidence", f"{prob:.1%}", 
                                 help="Raw probability output from the neural network")
                        st.metric("Detection Status", "POSITIVE" if is_detected else "NEGATIVE",
                                 help=f"Threshold: {PRESENCE_THRESHOLD:.0%}")
                    
                    with col2:
                        st.metric("Uncertainty (MC Dropout)", f"{uncertainty:.3f}",
                                 help="Epistemic uncertainty from Monte Carlo Dropout sampling")
                        st.metric("Confidence Level", 
                                 "High" if prob > 0.8 else "Medium" if prob > 0.6 else "Low" if prob > 0.4 else "Very Low",
                                 help="Qualitative assessment of prediction confidence")
                    
                    with col3:
                        # Calculate confidence interval
                        lower_bound = max(0, prob - 1.96 * uncertainty)
                        upper_bound = min(1, prob + 1.96 * uncertainty)
                        st.metric("95% CI Lower", f"{lower_bound:.1%}",
                                 help="Lower bound of 95% confidence interval")
                        st.metric("95% CI Upper", f"{upper_bound:.1%}",
                                 help="Upper bound of 95% confidence interval")
                    
                    # Confidence visualization
                    import plotly.graph_objects as go
                    conf_fig = go.Figure()
                    
                    # Add confidence bar
                    conf_fig.add_trace(go.Bar(
                        x=['Confidence'],
                        y=[prob],
                        marker=dict(color=disease_color),
                        text=[f"{prob:.1%}"],
                        textposition='outside',
                        name='Model Output'
                    ))
                    
                    # Add threshold line
                    conf_fig.add_hline(y=PRESENCE_THRESHOLD, line_dash="dash", 
                                      line_color="#00FFFF", 
                                      annotation_text=f"Detection Threshold ({PRESENCE_THRESHOLD:.0%})",
                                      annotation_font_color="#00FFFF")
                    
                    # Add uncertainty range
                    if uncertainty > 0:
                        conf_fig.add_trace(go.Scatter(
                            x=['Confidence', 'Confidence'],
                            y=[lower_bound, upper_bound],
                            mode='lines',
                            line=dict(color='rgba(255,255,255,0.3)', width=20),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                    
                    conf_fig.update_layout(
                        title=f"{disease_name} - Model Confidence",
                        yaxis=dict(title="Probability", range=[0, 1]),
                        height=300,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#E5E7EB'),
                        showlegend=False
                    )
                    st.plotly_chart(conf_fig, use_container_width=True)
                    
                    # Additional information
                    st.markdown(f"""
                    <div class="glass-card">
                        <h4 style="color: {disease_color};">Clinical Interpretation</h4>
                        <p style="color: #94A3B8; font-size: 13px;">
                            • <b>Confidence: {prob:.1%}</b> - Model's belief that {disease_name.lower()} is present<br>
                            • <b>Uncertainty: {uncertainty:.3f}</b> - Model's uncertainty about this prediction<br>
                            • <b>Status: {'DETECTED' if is_detected else 'NOT DETECTED'}</b> - Based on {PRESENCE_THRESHOLD:.0%} threshold<br>
                            • <b>Recommendation:</b> {'Expert review recommended for confirmation' if is_detected else 'Continue monitoring if clinical suspicion exists'}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Chart
            st.markdown("---")
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
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display metrics for tumor and stroke
                    if disease in ["tumor", "stroke"] and "lesion_metrics" in st.session_state:
                        metrics = st.session_state.lesion_metrics.get(disease, {})
                        if metrics:
                            vol = metrics.get("volume_mm3", 0)
                            centroid = metrics.get("centroid_mm", [0, 0, 0])
                            bbox_min = metrics.get("bbox_min_mm", [0, 0, 0])
                            bbox_max = metrics.get("bbox_max_mm", [0, 0, 0])
                            
                            st.markdown(f"""
                            <div style="margin-left: 20px; border-left: 2px solid {color}44; padding-left: 15px; margin-bottom: 20px;">
                                <p style="color: #E5E7EB; font-size: 13px; margin: 5px 0;">📏 <b>Volume:</b> {vol:,.1f} mm³ ({vol/1000:,.2f} mL)</p>
                                <p style="color: #E5E7EB; font-size: 13px; margin: 5px 0;">🎯 <b>Centroid:</b> ({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f}) mm</p>
                                <p style="color: #E5E7EB; font-size: 13px; margin: 5px 0;">📦 <b>Bounding Box (Min):</b> ({bbox_min[0]:.1f}, {bbox_min[1]:.1f}, {bbox_min[2]:.1f}) mm</p>
                                <p style="color: #E5E7EB; font-size: 13px; margin: 5px 0;">📦 <b>Bounding Box (Max):</b> ({bbox_max[0]:.1f}, {bbox_max[1]:.1f}, {bbox_max[2]:.1f}) mm</p>
                            </div>
                            """, unsafe_allow_html=True)
            
            # (Training Dashboard moved to dedicated TRAINING page)

        else:
            st.info("No analysis results available. Please upload and analyze a scan first.")
    
    elif st.session_state.current_page == 'training':
        # ========== TRAINING DASHBOARD PAGE ==========
        st.markdown("## 📈 Optimized 80-Epoch Curriculum Insights")
        st.markdown("---")

        tm = st.session_state.get("training_metrics", {})
        if not tm or not tm.get("epoch"):
            st.info("Training metrics not available in the current model. "
                    "Ensure you are using the optimized neurox_model.pth.")
        else:
            epochs_done = len(tm["epoch"])
            
            # --- MODEL HALL OF FAME (BEST SCORES) ---
            st.markdown("""
            <div class="glass-card">
                <h3 style="color: #00E5FF; margin-bottom: 20px;">🏆 Model Hall of Fame (All-Time Bests)</h3>
            </div>
            """, unsafe_allow_html=True)
            
            best_dice_tumor = max(tm.get("tumor_mean", [0]))
            best_et_tumor   = max(tm.get("tumor_et", [0]))
            best_dice_stroke = max(tm.get("stroke_dice", [0]))
            best_auc_alz    = max(tm.get("alz_auc", [0]))
            best_f1_alz     = max(tm.get("alz_f1", [0]))
            
            b1, b2, b3, b4 = st.columns(4)
            with b1:
                st.markdown(f"""
                <div class="glass-card" style="text-align: center; border-left: 4px solid #FF4444;">
                    <p style="color: #94A3B8; margin: 0; font-size: 11px; text-transform: uppercase; letter-spacing: 1px;">Best Tumor Mean Dice</p>
                    <h2 style="color: #FF4444; margin: 10px 0; font-family: 'Orbitron';">{best_dice_tumor:.4f}</h2>
                    <p style="color: #64748B; margin: 0; font-size: 10px;">ET Max: {best_et_tumor:.4f}</p>
                </div>
                """, unsafe_allow_html=True)
            with b2:
                st.markdown(f"""
                <div class="glass-card" style="text-align: center; border-left: 4px solid #4488FF;">
                    <p style="color: #94A3B8; margin: 0; font-size: 11px; text-transform: uppercase; letter-spacing: 1px;">Best Stroke Dice</p>
                    <h2 style="color: #4488FF; margin: 10px 0; font-family: 'Orbitron';">{best_dice_stroke:.4f}</h2>
                    <p style="color: #64748B; margin: 0; font-size: 10px;">Voxel-level Accuracy</p>
                </div>
                """, unsafe_allow_html=True)
            with b3:
                st.markdown(f"""
                <div class="glass-card" style="text-align: center; border-left: 4px solid #B67EFF;">
                    <p style="color: #94A3B8; margin: 0; font-size: 11px; text-transform: uppercase; letter-spacing: 1px;">Best Alzheimer AUC</p>
                    <h2 style="color: #B67EFF; margin: 10px 0; font-family: 'Orbitron';">{best_auc_alz:.4f}</h2>
                    <p style="color: #64748B; margin: 0; font-size: 10px;">Area Under Curve</p>
                </div>
                """, unsafe_allow_html=True)
            with b4:
                st.markdown(f"""
                <div class="glass-card" style="text-align: center; border-left: 4px solid #00FFFF;">
                    <p style="color: #94A3B8; margin: 0; font-size: 11px; text-transform: uppercase; letter-spacing: 1px;">Best Alzheimer F1</p>
                    <h2 style="color: #00FFFF; margin: 10px 0; font-family: 'Orbitron';">{best_f1_alz:.4f}</h2>
                    <p style="color: #64748B; margin: 0; font-size: 10px;">Harmonic Mean (P/R)</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            
            # --- TREND ANALYTICS ---
            import plotly.graph_objects as go
            def _dark_fig(title, y_title="Score"):
                fig = go.Figure()
                fig.update_layout(
                    title=dict(text=title, font=dict(family='Orbitron', size=16), x=0.05),
                    height=400,
                    plot_bgcolor="rgba(3,7,18,0.5)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#E5E7EB", family='Inter'),
                    xaxis=dict(title="Epoch", gridcolor="rgba(255,255,255,0.05)", range=[1, 80], zeroline=False),
                    yaxis=dict(title=y_title, gridcolor="rgba(255,255,255,0.05)", zeroline=False),
                    legend=dict(bgcolor="rgba(17,24,39,0.8)", bordercolor="rgba(0,229,255,0.2)", borderwidth=1),
                    margin=dict(t=60, b=40, l=60, r=20),
                    hovermode="x unified"
                )
                return fig

            def _add_phases(fig):
                # Phase 1: ALZ (1-20)
                fig.add_vrect(x0=1, x1=20.5, fillcolor="#B67EFF", opacity=0.08, layer="below", line_width=0)
                fig.add_annotation(x=10, y=0.95, text="PHASE 1: ALZ", showarrow=False, font=dict(color="#B67EFF", size=10, family='JetBrains Mono'))
                # Phase 2A: WARMUP (21-25)
                fig.add_vrect(x0=20.5, x1=25.5, fillcolor="#00E5FF", opacity=0.08, layer="below", line_width=0)
                fig.add_annotation(x=23, y=0.95, text="PHASE 2A: WARMUP", showarrow=False, font=dict(color="#00E5FF", size=10, family='JetBrains Mono'))
                # Phase 2B: FULL SEG (26-80)
                fig.add_vrect(x0=25.5, x1=80, fillcolor="#00FF88", opacity=0.08, layer="below", line_width=0)
                fig.add_annotation(x=53, y=0.95, text="PHASE 2B: FULL SEG", showarrow=False, font=dict(color="#00FF88", size=10, family='JetBrains Mono'))

            ep = tm["epoch"]
            
            # --- ROW 1: Segmentation Detailed ---
            st.markdown("### 🧬 Segmentation Accuracy Trends")
            fig_tumor = _dark_fig("Tumor Multi-Channel Dice (ET/NCR/ED)")
            _add_phases(fig_tumor)
            fig_tumor.add_trace(go.Scatter(x=ep, y=tm.get("tumor_et", []), name="Enhancing Tumor (ET)", line=dict(color="#FF4444", width=3)))
            fig_tumor.add_trace(go.Scatter(x=ep, y=tm.get("tumor_ncr", []), name="Necrotic Core (NCR)", line=dict(color="#FF8800", width=2, dash='dash')))
            fig_tumor.add_trace(go.Scatter(x=ep, y=tm.get("tumor_ed", []), name="Edema (ED)", line=dict(color="#FFFF00", width=2, dash='dot')))
            st.plotly_chart(fig_tumor, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                fig_mean = _dark_fig("Tumor Mean vs. Stroke Dice")
                _add_phases(fig_mean)
                fig_mean.add_trace(go.Scatter(x=ep, y=tm.get("tumor_mean", []), name="Tumor Mean", line=dict(color="#FF4444", width=3)))
                fig_mean.add_trace(go.Scatter(x=ep, y=tm.get("stroke_dice", []), name="Stroke Dice", line=dict(color="#4488FF", width=3)))
                st.plotly_chart(fig_mean, use_container_width=True)
            
            with c2:
                # --- ROW 2: Alzheimer Classification Trends ---
                fig_alz = _dark_fig("Alzheimer Primary Benchmarks")
                _add_phases(fig_alz)
                fig_alz.add_trace(go.Scatter(x=ep, y=tm.get("alz_auc", []), name="AUC-ROC", line=dict(color="#B67EFF", width=4)))
                fig_alz.add_trace(go.Scatter(x=ep, y=tm.get("alz_accuracy", []), name="Accuracy", line=dict(color="#FFFFFF", width=2, dash='dash')))
                fig_alz.add_trace(go.Scatter(x=ep, y=tm.get("alz_f1", []), name="F1 Score", line=dict(color="#00FFFF", width=2)))
                st.plotly_chart(fig_alz, use_container_width=True)

            # --- ROW 3: Precision & Recall ---
            st.markdown("### ⚖️ Detection Rigor (Alzheimer)")
            fig_pr = _dark_fig("Precision vs. Recall Stability")
            _add_phases(fig_pr)
            fig_pr.add_trace(go.Scatter(x=ep, y=tm.get("alz_precision", []), name="Precision", line=dict(color="#00FF88", width=3)))
            fig_pr.add_trace(go.Scatter(x=ep, y=tm.get("alz_recall", []), name="Recall (Sensitivity)", line=dict(color="#FF00FF", width=3)))
            st.plotly_chart(fig_pr, use_container_width=True)
            
            st.markdown("""
            <div class="glass-card" style="border-left-color: #00E5FF;">
                <h4 style="color: #00E5FF;">💡 Advanced Curriculum Analysis</h4>
                <p style="color: #94A3B8; font-size: 13px; line-height: 1.6;">
                    • <b>Phase 1 (Epochs 1-20):</b> Alzheimer Pre-training. Dedicated AlzheimerEncoder establishes base features.<br>
                    • <b>Phase 2A (Epochs 21-25):</b> Segmentation Warmup. Convolutions are trained with lower LR to stabilize spatial paths.<br>
                    • <b>Phase 2B (Epochs 26-80):</b> Full Segmentation. Transformer bottleneck unfrozen for global multi-disease context modeling.
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    elif st.session_state.current_page == 'visualization':
        # ========== VISUALIZATION PAGE ==========
        st.markdown("## 🧠 3D Visualization Laboratory")
        st.markdown("---")
        
        # FIX 7.3 — Validate all required session state keys before attempting viz
        required_keys = ["segmentation_results", "roi_metadata", "original_image", "affine", "spacing"]
        missing = [k for k in required_keys if k not in st.session_state or st.session_state[k] is None]
        if missing:
            st.error(f"⚠️ Missing analysis data: {missing}. Please re-run the analysis.")
            st.stop()
        
        if st.session_state.segmentation_results:
            st.markdown("""
            <div class="glass-card">
                <h3 style="color: #00E5FF; margin-bottom: 10px;">🌐 3D Brain Rendering</h3>
                <p style="color: #94A3B8; font-size: 13px;">Interactive 3D visualization of detected pathologies.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Patient-specific brain visualization (Otsu brain mask + separate lesion meshes)
            print("\n" + "="*60)
            print("🎬 Starting 3D visualization...")
            print(f"📊 Segmentations available: {list(st.session_state.segmentation_results.keys())}")
            print("="*60 + "\n")
            
            # ITERATE THROUGH EACH DETECTED DISEASE FOR SEPARATE VISUALIZATION
            detected_diseases = [d for d in st.session_state.detection_results["detected_diseases"] 
                                if d in st.session_state.segmentation_results]
            
            if not detected_diseases:
                st.info("No segmentable diseases detected (Alzheimer is presence-only).")
            
            for disease in detected_diseases:
                disease_name = DISEASE_COLORS[disease]["name"]
                disease_color = DISEASE_COLORS[disease]["hex"]
                
                st.markdown(f"### {disease_name} Visualization")
                
                # Filter segmentation results for just this disease
                single_disease_seg = {disease: st.session_state.segmentation_results[disease]}
                
                fig_3d = create_3d_visualization(
                    segmentations_roi=single_disease_seg,
                    roi_metadata=st.session_state.roi_metadata,
                    original_volume=st.session_state.original_image,
                    affine=st.session_state.affine,
                    spacing=st.session_state.spacing,
                    show_patient_brain=st.session_state.show_atlas,
                    show_heatmap=st.session_state.show_heatmap,
                    lesion_metrics=st.session_state.get("lesion_metrics")
                )
                
                if len(fig_3d.data) > 0:
                    st.plotly_chart(fig_3d, use_container_width=True, key=f"viz_{disease}")
                    
                    # Add detailed voxel stats
                    probs, binary = st.session_state.segmentation_results[disease]
                    # Note: These stats are ROI based, but give a sense of scale
                    if disease == "tumor" and probs.ndim == 4:
                        # Show stats for the channel used (likely 3 or 1)
                        # We re-calculate the 'strict' mask used in viz to show accurate count
                        pass 
                else:
                    st.warning(f"⚠️ Could not generate 3D mesh for {disease_name} (volume might be too small).")

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
                    # Check for API Key in Session State
                    api_key = st.session_state.get('groq_api_key', None)
                    
                    if not api_key:
                        st.warning("⚠️ No Groq API Key found. Using fallback template. Go to **Settings** to configure AI.")
                    
                    with st.spinner("✍️ Generating AI report..."):
                        st.session_state.report_text = generate_ai_report(
                            st.session_state.detection_results,
                            st.session_state.segmentation_results,
                            api_key
                        )
            
            if st.session_state.report_text:
                st.markdown("---")
                st.markdown(st.session_state.report_text)
                
                st.markdown("---")
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col2:
                    if st.button("💾 SAVE TEXT", use_container_width=True):
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

    elif st.session_state.current_page == 'settings':
        # ========== SETTINGS PAGE ==========
        st.markdown("## ⚙️ System Settings")
        st.markdown("---")
        
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #00E5FF; margin-bottom: 10px;">🔑 API Configuration</h3>
            <p style="color: #94A3B8; font-size: 13px;">Configure external AI services for enhanced reporting.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize key if not present
        if 'groq_api_key' not in st.session_state:
            st.session_state.groq_api_key = ""
        
        # API Key Input
        col1, col2 = st.columns([3, 1])
        with col1:
            new_key = st.text_input(
                "Groq API Key", 
                value=st.session_state.groq_api_key, 
                type="password",
                help="Enter your Groq API key (starts with gsk_)"
            )
        
        with col2:
            st.write("") # Spacer
            st.write("") 
            if st.button("💾 SAVE KEY", use_container_width=True):
                st.session_state.groq_api_key = new_key
                st.success("✅ API Key Saved!")
                
        if st.session_state.groq_api_key:
            st.info(f"✅ Active Key: {st.session_state.groq_api_key[:8]}...{st.session_state.groq_api_key[-4:]}")
        else:
            st.warning("⚠️ No API Key configured. AI reporting will be disabled.")
            
        # 2. Visualization Options (Restored)
        st.markdown("---")
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #00E5FF; margin-bottom: 10px;">🎨 Visualization Options</h3>
            <p style="color: #94A3B8; font-size: 13px;">Customize how 3D models and results are rendered.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.session_state.show_atlas = st.checkbox("Show Brain Surface Atlas", value=st.session_state.show_atlas,
                                                     help="Toggle the translucent brain shell reference.")
        with col_v2:
            st.session_state.show_heatmap = st.checkbox("Show Probability Heatmap", value=st.session_state.show_heatmap,
                                                       help="Overlay probability gradients on the segmentation.")

        # 3. Detection Config (Restored)
        st.markdown("---")
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #00E5FF; margin-bottom: 10px;">📊 Analysis Configuration</h3>
            <p style="color: #94A3B8; font-size: 13px;">Adjust sensitivity and detection parameters.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.session_state.confidence_threshold = st.slider(
            "Confidence Threshold", 
            0.0, 1.0, 
            st.session_state.confidence_threshold, 
            0.05,
            help="Minimum confidence required to flag a disease as detected."
        )
        st.caption(f"Current System Sensitivity: **{st.session_state.confidence_threshold:.0%}** (Lower = More Sensitive/More False Positives)")
            
        st.markdown("---")
        st.markdown("### 🛠️ System Info")
        st.code(f"""
        NeuroX Version: 1.5.0
        PyTorch: {torch.__version__}
        Device: {DEVICE}
        HD-BET: {'Available ✅' if HDBET_AVAILABLE else 'Not Found ❌'}
        """)


if __name__ == "__main__":
    run_streamlit_app()
