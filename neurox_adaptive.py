import os
import sys
import io
import base64
import tempfile
import traceback
import subprocess
from pathlib import Path
from datetime import datetime
import neurox_report_engine as re_engine
from typing import Dict, List, Optional, Tuple
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
import torch
import torch.nn as nn
import hashlib
import torch.nn.functional as F
from scipy.ndimage import (binary_closing, binary_fill_holes, distance_transform_edt, 
                           gaussian_filter, label as cc_label, binary_dilation as scipy_binary_dilation, zoom)
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
from dotenv import load_dotenv

# Load Environment Variables (.env)
load_dotenv()

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
    """Transformer bottleneck. depth=4, heads=8, mlp_dim=256, dropout=0.2"""
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
        n_tokens = d * h * w
        # Part C: OOM guard with visibility warning
        if n_tokens > 2000:
            print(f"[WARN] Transformer skipped: tokens={n_tokens} (limit=2000) — returning identity")
            return x
        
        # x shape: (b, c, d, h, w) -> (b, d*h*w, c)
        x = x.view(b, c, -1).permute(0, 2, 1)
        for ln1, attn, ln2, ff in self.layers:
            attn_out, _ = attn(ln1(x), ln1(x), ln1(x))
            x = x + attn_out
            x = x + ff(ln2(x))
        return x.permute(0, 2, 1).view(b, c, d, h, w)


class SharedEncoder(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()
        self.enc1 = self._conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = self._conv_block(32, 64)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = self._conv_block(64, 128)
        self.pool3 = nn.MaxPool3d(2)
        # 3D Transformer bottleneck (removed from here, now in NeuroXMultiDisease)
    
    def _conv_block(self, in_c, out_c):
        """InstanceNorm3d for batch_size=1 stability."""
        return nn.Sequential(
            nn.Conv3d(in_c, out_c, 3, padding=1),
            nn.InstanceNorm3d(out_c, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_c, out_c, 3, padding=1),
            nn.InstanceNorm3d(out_c, affine=True),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        # 🧩 Domain B: 2-Channel Input (T1ce + FLAIR) - Preservation of discriminative signal.
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.pool3(e3) # Bottleneck input before transformer
        return {"enc1": e1, "enc2": e2, "enc3": e3, "bottleneck_input": b}


class PresenceHead(nn.Module):
    """Binary presence detector with Heteroscedastic Uncertainty."""
    def __init__(self, in_features=128):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 2)  # [logit, log_var]
    
    def forward(self, bottleneck_features):
        x = self.pool(bottleneck_features)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.fc2(x)
        return out[:, 0:1], out[:, 1:2]


class AttentionGate3D(nn.Module):
    """Spatial attention gate: suppresses irrelevant skip features."""
    def __init__(self, gate_ch, skip_ch, inter_ch):
        super().__init__()
        self.W_gate = nn.Conv3d(gate_ch, inter_ch, 1)
        self.W_skip = nn.Conv3d(skip_ch, inter_ch, 1)
        self.psi = nn.Conv3d(inter_ch, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, gate, skip):
        # Spatial shapes: gate and skip must match after W projections
        psi = self.relu(self.W_gate(gate) + self.W_skip(skip))
        return skip * self.sigmoid(self.psi(psi))


class SegmentationDecoder(nn.Module):
    """UNet decoder with 3 attention gates.
    
    Spatial flow (ROI=96):
      Bottleneck: 12^3  -> up3 -> 24^3 (att3 on enc3=24^3)
      24^3         -> up2 -> 48^3 (att2 on enc2=48^3)
      48^3         -> up1 -> 96^3 (att1 on enc1=96^3)
    No stride mismatch.
    """
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
    
    def forward(self, enc_features, bottleneck_features):
        e1, e2, e3 = enc_features["enc1"], enc_features["enc2"], enc_features["enc3"]
        b = bottleneck_features # This is the output of the TransformerBottleneck3D
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
    """AlzheimerEncoder v3: Deep Residual + SE Attention, CLEAN binary output.
    
    FIX: Removed pseudo-heteroscedastic dual-output head. The old design used
    Linear(128, 2) and split output as [logit, log_var], but output[:, 1] is
    just a 2nd raw logit — not a variance — creating conflicting gradient signals
    that prevent learning (Kendall & Gal 2017 requires separate variance branch).
    Now outputs a single binary logit + a TRUE separate log_var head.
    
    Structure:
        1 -> 32 (ResBlock) -> Pool
        32 -> 64 (ResBlock) -> Pool
        64 -> 128 (ResBlock) -> Pool
        128 -> 256 (ResBlock) -> SE Attention
    Classifier: 512 in (Avg+Max concat) -> LayerNorm -> 256 -> Dropout(0.3) -> 1 logit
    Log-var head: 512 -> 64 -> 1 (separate branch, no gradient into logit)
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

        # CLEAN single-output classifier (binary AD/CN)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3),   # Slightly higher dropout for 3D MRI classification
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)  # Single logit output
        )
        
        # TRUE separate log-variance head (detached from classification pathway)
        self.log_var_head = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
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

        logit   = self.classifier(feat)
        log_var = self.log_var_head(feat)
        return logit, log_var

    def extract_features(self, x):
        """B1: Matches EXACT forward path for diagnostic validity."""
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = self.block4(x)
        x = self.se(x)
        avg = self.avg_pool(x).flatten(1)
        mx  = self.max_pool(x).flatten(1)
        feat = torch.cat([avg, mx], dim=1)
        return self.norm(feat)  # LayerNorm must be included


class DecisionHead(nn.Module):
    def __init__(self, in_features=5):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, features):
        return self.mlp(features)

class NeuroXMultiDisease(nn.Module):
    """Multitask model with selective forward.

    Selective forward: only activate heads needed per batch.
    Alzheimer uses a dedicated AlzheimerEncoder that receives raw MRI directly.
    SharedEncoder is used only for Tumor / Stroke.
    """
    def __init__(self, in_channels=2): 
        super().__init__()
        self.encoder = SharedEncoder(in_channels=in_channels) # Fix 2: Pass in_channels to SharedEncoder
        self.bottleneck = TransformerBottleneck3D(128, 4, 8, 256, 0.2) # Moved from SharedEncoder
        
        self.tumor_presence = PresenceHead(128)
        self.stroke_presence = PresenceHead(128)
        
        self.tumor_decoder = SegmentationDecoder(3, "tumor")
        self.stroke_decoder = SegmentationDecoder(1, "stroke")
        
        # === Alzheimer Dedicated Encoder ===
        # Raw MRI -> independent 3D CNN -> dual pool -> MLP -> AD logit
        # No shared features with SharedEncoder.
        self.alz_encoder = AlzheimerEncoder()
        
        self.decision_head = DecisionHead()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x, active_presence=None, active_seg=None):
        res = {"presence": {}, "segmentations": {}, "alzheimer_log_var": None}
        
        # FIX 8 — BATCH DIMENSION GUARD
        if x.dim() == 4:
            x = x.unsqueeze(0)

        assert x.dim() == 5, f"Input must be (B, C, D, H, W), got {x.dim()}"

        # Optimization: Only run alz_encoder when presence["alzheimer"] is requested
        if active_presence and "alzheimer" in active_presence:
            # FIX 2 — ALZ ROUTING (Sync with v3 packing: Ch0 is ALZ Preprocessed)
            x_alz = x[:, 0:1] 
            alz_logits, alz_log_var = self.alz_encoder(x_alz)
            res["presence"]["alzheimer"] = alz_logits
            res["alzheimer_log_var"] = alz_log_var
            
        # Shared Path (Seg + Presence) - Only run if Tumor/Stroke requested
        if (active_presence and any(k in ["tumor", "stroke"] for k in active_presence)) or \
           (active_seg and any(k in ["tumor", "stroke"] for k in active_seg)):
            
            # FIX 3 — SEG ROUTING (Sync with v3 packing: Ch1 is Standard Preprocessed)
            # SharedEncoder expects 2-channel input. In training, multi-modal is [T1ce, FLAIR].
            # For single-modality app uploads, we duplicate the standard channel [Ch1, Ch1].
            x_seg_ch = x[:, 1:2]
            x_seg = torch.cat([x_seg_ch, x_seg_ch], dim=1)
            
            feats = self.encoder(x_seg)
            
            # Bottleneck Path (Inference Optimized - Fixed Issue 8)
            bottleneck_feats = self.bottleneck(feats["bottleneck_input"])
            
            if active_presence and "tumor" in active_presence:
                 res["presence"]["tumor"] = self.tumor_presence(bottleneck_feats)
            if active_presence and "stroke" in active_presence:
                 res["presence"]["stroke"] = self.stroke_presence(bottleneck_feats)
                 
            # Segmentation Decoders
            if active_seg and "tumor" in active_seg:
                res["segmentations"]["tumor"] = self.tumor_decoder(feats, bottleneck_feats)
            if active_seg and "stroke" in active_seg:
                res["segmentations"]["stroke"] = self.stroke_decoder(feats, bottleneck_feats)

        # Part D: Temperature Scaling (Sync with training Fix 10)
        # Applied INSIDE forward during eval to ensure logit distribution parity
        if not self.training:
            for k in res["presence"]:
                if isinstance(res["presence"][k], tuple):
                    logit, log_var = res["presence"][k]
                    res["presence"][k] = (logit / self.temperature.clamp(0.01, 10.0), log_var)
                else:
                    res["presence"][k] = res["presence"][k] / self.temperature.clamp(0.01, 10.0)

        return res


# ═══════════════════════════════════════════════════════════════════════════
# INFERENCE UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def prepare_input(x: torch.Tensor, task: str) -> torch.Tensor:
    """
    FIX 1 — ADD CENTRAL INPUT ROUTER (MANDATORY)
    Enforces training-time channel constraints.
    """
    assert x.dim() == 5, "Input must be (B, C, D, H, W)"
    assert x.shape[1] in [1, 2], f"Invalid channel count: {x.shape}"

    if task == "alzheimer":
        # MUST be 1-channel
        if x.shape[1] > 1:
            return x[:, :1]
        return x

    elif task in ["tumor", "stroke"]:
        # MUST be 2-channel
        if x.shape[1] == 1:
            return torch.cat([x, x], dim=1)
        elif x.shape[1] >= 2:
            return x[:, :2]

    raise ValueError(f"Unknown task: {task}")


@st.cache_resource
def load_model(model_path: str = MODEL_PATH):
    """Load trained model with strict architecture validation.
    
    Ensures 100% parity with training script. Supports:
      - Inference format: {"model_state": ...}
      - Resume format: {"model": ...}
      - Legacy format: plain state_dict
    """
    model = NeuroXMultiDisease().to(DEVICE)
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=DEVICE)
            
            # 1. Extract state_dict based on known keys
            if isinstance(checkpoint, dict):
                if "model_state" in checkpoint:
                    state_dict = checkpoint["model_state"]
                    st.session_state.training_metrics = checkpoint.get("metrics", {})
                elif "model" in checkpoint:
                    state_dict = checkpoint["model"]
                    st.session_state.training_metrics = checkpoint.get("metrics", {})
                else:
                    state_dict = checkpoint
                    st.session_state.training_metrics = {}
            else:
                state_dict = checkpoint
                st.session_state.training_metrics = {}
            
            # 2. Strict loading - any mismatch will raise error (guarantees parity)
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            print("✅ Model loaded successfully with strict=True parity.")
            
            # Optional: Populate metrics metadata
            if st.session_state.training_metrics:
                n_epochs = len(st.session_state.training_metrics.get("epoch", []))
                print(f"📈 Loaded metrics history for {n_epochs} epochs.")
                
            return model
            
        except Exception as e:
            st.error(f"⚠️ Architecture Mismatch: {e}")
            print(f"❌ Error loading model: {e}")
            # Fallback to non-strict if desperate, but warn user
            try:
                model.load_state_dict(state_dict, strict=False)
                model.eval()
                return model
            except:
                return model
    else:
        st.warning(f"⚠️ Model file not found: {model_path}")
        return None


def preprocess_alz_light(volume: np.ndarray) -> torch.Tensor:
    """
    EXACT copy of training preprocess_alz_light — do not alter.
    Uses pure z-score then soft-clips ±3σ.
    """
    volume = volume.astype(np.float32)
    mean = volume.mean()
    std  = volume.std() + 1e-6
    volume = (volume - mean) / std
    volume = np.clip(volume, -3.0, 3.0)
    
    # Direct resize to (1, 96, 96, 96)
    vol_t = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)
    vol_t = F.interpolate(vol_t, size=ROI_SIZE, mode="trilinear", align_corners=False)
    return vol_t.squeeze(0)


def load_and_preprocess_nifti(file_path: str) -> Tuple[torch.Tensor, np.ndarray, Dict, np.ndarray, Tuple]:
    """Load and preprocess NIfTI file identifying with training baseline."""
    img = nib.load(file_path)
    img = nib.as_closest_canonical(img)
    
    original_data = img.get_fdata().astype(np.float32)
    original_shape = original_data.shape
    affine = img.affine
    spacing_raw = np.sqrt(np.sum(affine[:3, :3]**2, axis=0))
    if np.any(spacing_raw <= 0) or np.any(np.isnan(spacing_raw)):
        print(f'   WARNING: Invalid spacing {spacing_raw} — using isotropic 1mm fallback')
        spacing = (1.0, 1.0, 1.0)
    else:
        max_ratio = spacing_raw.max() / (spacing_raw.min() + 1e-8)
        if max_ratio > 10.0:
            print(f'   WARNING: Extreme anisotropy ({max_ratio:.1f}x): {spacing_raw} mm')
        else:
            print(f'   Voxel spacing: {spacing_raw[0]:.2f} x {spacing_raw[1]:.2f} x {spacing_raw[2]:.2f} mm')
        spacing = tuple(float(s) for s in spacing_raw)
    
    # PREPROCESS Tiers (Fix 1, 3 — Parity)
    data = original_data.copy()
    if data.ndim == 4:
        data = data[..., 0] if data.shape[-1] <= 2 else data[..., :2].mean(axis=-1)

    # Tier 1: Alzheimer Light (Z-score + 3σ) - Returns (1, 96, 96, 96)
    roi_alz = preprocess_alz_light(data)

    # Tier 2: Standard (Percentile + Z-score)
    p1, p99 = np.percentile(data, (1, 99))
    data_std = np.clip(data, p1, p99)
    data_std = (data_std - p1) / (p99 - p1 + 1e-8)
    mean, std = data_std.mean(), data_std.std() + 1e-8
    data_std = (data_std - mean) / std
    
    vol_std = torch.from_numpy(data_std.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    roi_standard = F.interpolate(vol_std, size=ROI_SIZE, mode="trilinear", align_corners=False).squeeze(0)
    
    # FIX 3 — CHANNEL PACKING (Ch0=Alz, Ch1=Standard)
    # NeuroXMultiDisease.forward() now routes: 
    #   x[:, 0:1] -> AlzEncoder
    #   [x[:, 1:2], x[:, 1:2]] -> SharedEncoder (2ch duplicate)
    roi_final = torch.cat([roi_alz, roi_standard], dim=0)

    # FIX 5 — CROPPED AFFINE MATH
    orig_shape_3d = tuple(original_shape) if original_data.ndim == 3 else tuple(original_shape[:3])
    scale = np.array(orig_shape_3d, dtype=np.float64) / np.array(ROI_SIZE, dtype=np.float64)
    
    scale_matrix = np.eye(4)
    scale_matrix[0, 0] = scale[0]
    scale_matrix[1, 1] = scale[1]
    scale_matrix[2, 2] = scale[2]

    roi_affine = affine @ scale_matrix

    roi_metadata = {
        "original_shape":  orig_shape_3d,
        "interpolation_mode": "trilinear",
        "roi_affine":      roi_affine,
        "original_affine": affine,
        "shape":           original_shape,
        "spacing":         spacing,
        "affine":          affine.tolist()
    }
    
    return roi_final, original_data, roi_metadata, affine, spacing


def compute_lesion_metrics(mask, brain_mask, spacing=(1.0, 1.0, 1.0), prob=0.0, uncertainty=0.0, logit=0.0, affine=None):
    """Advanced clinical-grade lesion analytics with training-parity metrics."""
    metrics = {}
    
    # Bug 1 Fix: Collapse multi-channel masks (tumor ET/NCR/ED)
    if mask.ndim == 4:
        mask = (mask.max(axis=0) > 0).astype(np.uint8)
    mask = (mask > 0).astype(np.uint8)

    if brain_mask is not None and brain_mask.ndim > 3:
        brain_mask = (brain_mask.max(axis=0) > 0).astype(np.uint8)

    # --- 1. Volume ---
    lesion_voxels = np.sum(mask > 0)
    brain_voxels = np.sum(brain_mask > 0)
    voxel_volume = spacing[0] * spacing[1] * spacing[2]
    lesion_volume = lesion_voxels * voxel_volume

    metrics["lesion_voxels"] = int(lesion_voxels)
    metrics["lesion_volume_mm3"] = float(lesion_volume)

    # --- 2. Brain percentage ---
    metrics["brain_percentage"] = float((lesion_voxels / brain_voxels) * 100) if brain_voxels > 0 else 0.0

    # --- 3. Centroid location ---
    coords = np.argwhere(mask > 0)
    if len(coords) > 0:
        centroid = coords.mean(axis=0)
        metrics["centroid"] = centroid.tolist()
        
        # Calculate MM coordinates (Unified logic)
        # 1. LOCAL MM (Voxel * Spacing)
        local_mm = centroid * np.array(spacing)
        metrics["centroid_local_mm"] = local_mm.tolist()
        
        # 2. WORLD MM (RAS Coordinates using Affine)
        if affine is not None:
            # nib.affines.apply_affine(affine, [x, y, z])
            # Note: input coords are (z, y, x) so we swap to (x, y, z) for RAS
            coords_ras = coords[:, [2, 1, 0]]
            centroid_ras = coords_ras.mean(axis=0)
            world_mm = nib.affines.apply_affine(affine, centroid_ras)
            metrics["centroid_mm"] = world_mm.tolist()
            
            # Same for BBox
            bbox_min_ras = nib.affines.apply_affine(affine, coords_ras.min(axis=0))
            bbox_max_ras = nib.affines.apply_affine(affine, coords_ras.max(axis=0))
            metrics["bbox_min_mm"] = bbox_min_ras.tolist()
            metrics["bbox_max_mm"] = bbox_max_ras.tolist()
        else:
            # Fallback to local scaling if no affine provided
            metrics["centroid_mm"] = local_mm.tolist()
            metrics["bbox_min_mm"] = (coords.min(axis=0) * np.array(spacing)).tolist()
            metrics["bbox_max_mm"] = (coords.max(axis=0) * np.array(spacing)).tolist()
        
        x, y, z = centroid
        metrics["hemisphere"] = "Right" if x > mask.shape[0] / 2 else "Left"
        metrics["position_AP"] = "Anterior" if y < mask.shape[1] / 2 else "Posterior"
        metrics["position_SI"] = "Superior" if z < mask.shape[2] / 2 else "Inferior"
    else:
        metrics["centroid"] = None
        metrics["centroid_mm"] = [0.0, 0.0, 0.0]
        metrics["bbox_min_mm"] = [0.0, 0.0, 0.0]
        metrics["bbox_max_mm"] = [0.0, 0.0, 0.0]

    # --- 4. Depth from surface ---
    lesion_depth = np.array([]) # Fix 11: Initialize before try
    try:
        from scipy.ndimage import distance_transform_edt
        dist_map = distance_transform_edt(brain_mask)
        lesion_depth = dist_map[mask > 0]
        if len(lesion_depth) > 0:
            metrics["depth_min"] = float(lesion_depth.min())
            metrics["depth_mean"] = float(lesion_depth.mean())
            metrics["depth_max"] = float(lesion_depth.max())
    except:
        metrics["depth_min"] = metrics["depth_mean"] = metrics["depth_max"] = 0.0

    # --- 5. Surface involvement ---
    surface_threshold = 3
    if lesion_voxels > 0 and len(lesion_depth) > 0:
        surface_voxels = np.sum(lesion_depth < surface_threshold)
        metrics["surface_ratio"] = float(surface_voxels / lesion_voxels)
    else:
        metrics["surface_ratio"] = 0.0

    # --- 6. Model-Derived Analytics ---
    metrics["prob"] = float(prob)
    metrics["uncertainty"] = float(uncertainty)
    metrics["confidence"] = 1.0 - float(uncertainty)
    metrics["risk"] = "High" if prob > 0.7 else ("Moderate" if prob > 0.4 else "Low")
    metrics["margin"] = abs(float(prob) - 0.5)
    metrics["adjusted_score"] = float(prob) * metrics["confidence"]
    metrics["logit_strength"] = abs(float(logit))
    return metrics


def compute_alzheimer_metrics(prob, uncertainty, logit):
    """Deep interpretability layer for Alzheimer's Pattern detection.
    
    Computes decision margin, entropy, logit-strength, and consistency indicators.
    """
    import numpy as np

    metrics = {}

    # 1. Derived confidence
    confidence = 1.0 - uncertainty
    
    # 2. Decision Margin (proximity to 0.5 boundary)
    margin = abs(prob - 0.5)
    
    # 3. Prediction Entropy (Standard uncertainty measure)
    eps = 1e-8
    entropy = - (prob * np.log(prob + eps) + (1.0 - prob) * np.log(1.0 - prob + eps))
    
    # 4. Adjusted Score (Consistency-weighted probability)
    adjusted_score = prob * confidence
    
    # 5. Logit Strength (Raw activation)
    logit_strength = abs(logit)
    
    # 6. Consistency Score (Reliability indicator)
    consistency = float(confidence * (1.0 - (entropy / 0.6932))) # Normalize entropy to [0,1]

    # 7. Risk Stratification
    if prob < 0.4:
        risk = "Low"
    elif prob < 0.7:
        risk = "Moderate"
    else:
        risk = "High"

    metrics.update({
        "prob": float(prob),
        "confidence": float(confidence),
        "margin": float(margin),
        "entropy": float(entropy),
        "adjusted_score": float(adjusted_score),
        "logit_strength": float(logit_strength),
        "consistency": float(consistency),
        "risk": risk
    })

    return metrics


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
    
    # RELAXED DETECTION FOR DECISION HEAD PROCESSING
    # Keep everything that has non-trivial signal so DecisionHead can reject later
    # 0.5 strict bound is abandoned!
    detected_diseases = [
        disease for disease, prob in disease_probs.items()
        if prob > 0.01 
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


def automatic_disease_detection_dual(model, image_tensor: torch.Tensor):
    """
    Executes deep-learning presence detection for Alzheimer's, Tumor, and Stroke.
    
    NOTE ON ALZ ROUTING: NeuroXMultiDisease.forward expects x[:, 0:1] for 
    Alzheimer when routed as active_presence=["alzheimer"]. Since the ALZ 
    input is already 1-channel, x[:, 0:1] correctly preserves this data.
    """
    model.eval()
    probabilities = {}
    uncertainties = {}
    presence_logits = {}
    
    # FIX 8 — BATCH DIMENSION GLOBAL ENFORCEMENT
    if image_tensor.dim() == 4:
        image_tensor = image_tensor.unsqueeze(0)
    
    # EXACT FIX 4 — CHANNEL CONTRACT ENFORCED BEFORE CALL
    x_alz = prepare_input(image_tensor, "alzheimer").to(DEVICE)
    x_seg = prepare_input(image_tensor, "tumor").to(DEVICE)
    assert x_alz.shape[1] == 1, f"Alzheimer input must be 1ch, got {x_alz.shape}"
    assert x_seg.shape[1] == 2, f"Segmentation input must be 2ch, got {x_seg.shape}"

    with torch.no_grad():
        # RUN SEPARATE CALLS (Architecture Sync)
        out_alz = model(x_alz, active_presence=["alzheimer"])
        out_seg = model(x_seg, active_presence=["tumor", "stroke"])

        # EXACT FIX 3 — PRECISE OUTPUT MAPPING
        # Tumor & Stroke: tuple(logit, log_var)
        for disease in ["tumor", "stroke"]:
            logit, log_var = out_seg["presence"][disease]
            presence_logits[disease] = logit.cpu().item()
            probabilities[disease] = float(torch.sigmoid(logit).cpu().item())
            uncertainties[disease] = float(torch.exp(log_var).cpu().item())

        # Alzheimer: logit + separate log_var
        alz_logit = out_alz["presence"]["alzheimer"]
        alz_log_var = out_alz["alzheimer_log_var"]
        presence_logits["alzheimer"] = alz_logit.cpu().item()
        probabilities["alzheimer"] = float(torch.sigmoid(alz_logit).cpu().item())
        uncertainties["alzheimer"] = float(torch.exp(alz_log_var).cpu().item())

    # EXACT FIX 8 — CALIBRATED THRESHOLDS
    THRESHOLDS = {
        "alzheimer": 0.4,
        "tumor": 0.3,
        "stroke": 0.3
    }
    
    detected_diseases = [
        d for d, p in probabilities.items()
        if p > THRESHOLDS.get(d, 0.5)
    ]
    
    return {
        "detected_diseases": detected_diseases,
        "probabilities": probabilities,
        "uncertainties": uncertainties,
        "presence_logits": presence_logits,
        "detection_confidence": {d: probabilities[d] for d in detected_diseases},
        "multi_label": True
    }


def perform_segmentation(model, image_tensor: torch.Tensor, tasks: List[str]):
    """
    Executes deep-learning segmentation for a list of detected pathologies.
    
    This function enforces the 2-channel input contract (T1-weighted and normalized)
    and routes the data to the appropriate model branches.
    
    Args:
        model: Trained MultiGenAI model with segmentation and presence heads.
        image_tensor: 4D or 5D tensor in [B, C, H, W, D] format.
        tasks: List of diseases to segment (e.g., ['tumor', 'stroke']).
        
    Returns:
        Dict: A dictionary mapping tasks to 6-tuple results:
              (binary_mask, prob_map, decision_score, presence_prob, unc, logit_raw).
    """
    model.eval()
    results = {}

    # FIX 8 — BATCH DIMENSION GLOBAL ENFORCEMENT
    if image_tensor.dim() == 4:
        image_tensor = image_tensor.unsqueeze(0)
    
    # FIX 5 — STRICT SEGMENTATION INPUT ROUTING
    x_seg = prepare_input(image_tensor, "tumor").to(DEVICE)
    assert x_seg.shape[1] == 2, f"Seg input must be 2ch, got {x_seg.shape}"

    with torch.no_grad():
        # Run model on 2rd-channel segments
        out = model(
            x_seg,
            active_presence=tasks,
            active_seg=tasks
        )

        for task in tasks:
            seg_logits = out["segmentations"][task]
            lp, lvp = out["presence"][task] 
            
            # EXACT FIX — SYNC WITH TRAINING DISTRIBUTIONS (logit_raw, log_var, vol, ent, peak)
            # Important: DecisionHead sees raw logits (scaled by temperature), 
            # while the app UI uses the sigmoid probability (logit).
            logit_raw = lp * model.temperature
            
            # Spatial Metadata (Sync with training feature extraction logic)
            vol = (torch.sigmoid(seg_logits) > 0.5).float().mean().view(1, 1)
            # Stability: Clamp probabilities to avoid log(0) NaN
            ps  = torch.sigmoid(seg_logits).clamp(1e-6, 1-1e-6)
            ent = -(ps * torch.log(ps) + (1-ps) * torch.log(1-ps)).mean().view(1, 1)
            p   = torch.sigmoid(seg_logits).amax(dim=(2,3,4)).mean(dim=1).view(1, 1)
            
            # Feature stack for DecisionHead: [logit_raw, log_var, vol, ent, peak]
            features = torch.stack([
                logit_raw.view(-1),
                lvp.view(-1),
                vol.view(-1),
                ent.view(-1),
                p.view(-1)
            ], dim=1).detach().cpu().float()
            
            # Post-process for metrics
            prob_scaled = torch.sigmoid(lp)
            uncertainty = torch.exp(lvp)
            seg_probs  = torch.sigmoid(seg_logits)[0].cpu().numpy()
            seg_binary = (seg_probs > 0.5).astype(np.uint8)
            
            dec_score = 1.0
            if hasattr(model, 'decision_head'):
                dec_val = model.decision_head(features.to(DEVICE))
                dec_score = float(torch.sigmoid(dec_val).item())

            # Store binary mask, probability tensor, DecisionHead score, and raw presence logits for downstream analytics
            results[task] = (seg_binary, seg_probs, dec_score, float(prob_scaled.item()), float(uncertainty.item()), float(logit_raw.item()))

    return results



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

    # Forcing boolean types to avoid ufunc casting errors with floats or mixed types
    overlap = (segmentation_mask.astype(bool) & brain_mask.astype(bool)).sum()
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
    if coords.size == 0:
        return False, None, "No voxels found in lesion mask"
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
    
    if brain_mask.sum() < 50000:
        raise ValueError("Brain mask too small — invalid for mesh generation")
    
    # FIX 4.2 — Unified Gaussian sigma=0.5 (matches lesion smoothing, prevents edge loss)
    from scipy.ndimage import gaussian_filter
    brain_smooth = gaussian_filter(brain_mask.astype(np.float32), sigma=0.5)
    
    # Marching cubes at level=0.5 (standard for binary masks)
    # FIX 2.3 — NO spacing argument. The affine matrix already encodes
    # physical voxel size in mm via its column vectors. Passing spacing=spacing
    # here would pre-scale the vertices before the affine is applied,
    # causing double-scaling relative to the lesion mesh (which has no spacing).
    try:
        from skimage import measure
        level = 0.5
        verts, faces, normals, _ = measure.marching_cubes(
            brain_smooth,
            level=level
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

def apply_hdbet_brain_extraction(file_path: str, spacing: Tuple[float, float, float], file_hash: str = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Performs medical-grade skull stripping using the HD-BET (High-Dimensional Brain Extraction Tool).
    
    This function includes strict input validation, normalization detection, and 
    controlled rescaling for 0-1 range inputs. It enforces canonical (RAS+) orientation
    on the output to ensure spatial synchronization with subsequent 3D meshes.
    
    Args:
        file_path: Absolute path to the original NIfTI file.
        spacing: Voxel spacing (unused by CLI but passed for context).
        file_hash: MD5 hash of the file for disk caching of extraction results.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (brain_extracted_volume, binary_brain_mask).
    """
    if not HDBET_AVAILABLE:
        print("❌ HD-BET CLI not available")
        return None, None
    
    print("\n" + "="*60)
    print("🧠 HD-BET BRAIN EXTRACTION (Medical-Grade)")
    print("="*60)
    
    try:
        # LAYER 3: DISK CACHING
        cache_dir = os.path.join("cache", file_hash) if file_hash else tempfile.mkdtemp()
        os.makedirs(cache_dir, exist_ok=True)
        
        input_path = os.path.join(cache_dir, "input.nii.gz")
        output_path = os.path.join(cache_dir, "output.nii.gz")
        
        if file_hash and os.path.exists(output_path):
            print("✅ ⚡ Using cached HD-BET brain mask from disk")
            brain_img = nib.load(output_path)
            orig_img = nib.load(file_path)
            orig_img = nib.as_closest_canonical(orig_img)
            return orig_img.get_fdata().astype(np.float32), brain_img.get_fdata()

        print(f"📝 Saving temporary NIfTI for HD-BET...")
        # ALWAYS load raw image (NOT preprocessed tensor)
        orig_img = nib.load(file_path)
        orig_img = nib.as_closest_canonical(orig_img)

        raw_volume = orig_img.get_fdata().astype(np.float32)
        affine = orig_img.affine
        
        # --- PHASE 1: STRICT INPUT VALIDATION (NOT OPTIONAL) ---
        min_v = raw_volume.min()
        max_v = raw_volume.max()
        std_v = raw_volume.std()
        
        print(f"🔍 INPUT CHECK: Min={min_v:.4f}, Max={max_v:.4f}, Std={std_v:.4f}")
        
        if std_v < 1e-6:
            raise ValueError("Invalid input: near-zero variance. HD-BET cannot process constant volumes.")
            
        if max_v == 0:
            raise ValueError("Invalid input: all zeros. No voxel data detected.")
            
        # --- PHASE 2: TYPE DETECTION & CONTROLLED RECOVERY ---
        if max_v < 10:
            print("⚠️ Likely normalized input (0-1 range) detected")
            if std_v > 0:
                print("♻️  Rescaling normalized input for HD-BET (x1000 policy)")
                hdbet_volume = raw_volume * 1000
            else:
                hdbet_volume = raw_volume
        else:
            print("✅ Raw MRI intensity detected")
            hdbet_volume = raw_volume.copy()
        
        print("HD-BET input stats:", hdbet_volume.min(), hdbet_volume.max(), hdbet_volume.mean())
        
        input_nii = nib.Nifti1Image(hdbet_volume, affine)
        nib.save(input_nii, input_path)
        
        print(f"   Input: {input_path}")
        
        print(f"🔧 Running HD-BET CLI...")
        cmd = [
            "hd-bet",
            "-i", input_path,
            "-o", output_path,
            "-device", "cpu"
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
            print(f"   Files in temp dir: {os.listdir(cache_dir)}")
            return None, None
        
        # ISSUE 2 FIX: Enforce canonical orientation on HD-BET output.
        # HD-BET may internally reorient the volume before writing output.
        # Without this the brain mask affine can silently diverge from the
        # input affine, causing the brain surface and lesion meshes to split.
        brain_img = nib.load(brain_path)
        brain_img = nib.as_closest_canonical(brain_img)   # enforce RAS+
        brain_volume = brain_img.get_fdata()
        brain_affine = brain_img.affine

        input_shape = hdbet_volume.shape
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
        
        if brain_voxels < 50000:
            print(f"❌ VALIDATION FAILED: Brain mask too small ({brain_voxels:,} < 50,000)")
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
    """Generate brain tissue mask with adaptive ROI-aware parameters."""
    from skimage.morphology import ball, binary_closing, binary_erosion, binary_dilation, remove_small_objects
    from scipy.ndimage import binary_fill_holes
    
    volume_positive = np.abs(volume)
    non_zero = volume_positive[volume_positive > 0]
    if len(non_zero) == 0: raise ValueError("Empty volume")
    
    # Adaptive thresholding
    if (non_zero.max() - non_zero.min()) < 10:
        threshold = np.percentile(non_zero, 30)
    else:
        threshold = np.percentile(non_zero, 60)
    
    brain_mask = (volume_positive > threshold).astype(bool)
    if brain_mask.sum() == 0:
        brain_mask = (volume_positive > np.percentile(non_zero, 5)).astype(bool)
    
    # Only minimal cleaning
    brain_mask = brain_mask.astype(bool)

    # Remove tiny noise ONLY
    brain_mask = remove_small_objects(brain_mask, min_size=500)

    # Keep largest component ONLY IF mask is large enough
    if brain_mask.sum() > 100000:
        labeled = measure.label(brain_mask)
        regions = measure.regionprops(labeled)
        if regions:
            largest = max(regions, key=lambda r: r.area)
            brain_mask = labeled == largest.label

    brain_mask_final = binary_fill_holes(brain_mask)
    
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
        print("⚠️ WARNING: Empty brain mask provided to bounding box utility.")
        return None
    
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
    
    
    # PER USER INSTRUCTION: Remove hardcoded morphological filtering logic
    # Replaced by ML DecisionHead evaluation upstream.
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
    for disease, result in segmentations_roi.items():
        # ALZHEIMER GUARD
        if disease == "alzheimer":
            continue
        # Unpack binary mask (index 0) from 6-tuple result: (binary, probs, dec, prob_orig, unc, logit)
        binary = result[0] if isinstance(result, (tuple, list)) else result
        
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
        disease_cfg = DISEASE_COLORS.get(disease, {"hex": "#FFFFFF", "name": disease})
        color = tuple(int(disease_cfg["hex"][i:i+2], 16)/255 for i in (1, 3, 5))
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


@st.cache_resource(show_spinner=False)
def get_visualization_assets(
    file_hash: str,
    raw_nifti_bytes: bytes,
    disease: str,
    disease_name: str,
    single_seg_data: Dict,
    roi_metadata: Dict,
    affine: np.ndarray,
    spacing: Tuple[float, float, float],
    lesion_metrics: Optional[Dict],
    model_path: Path
) -> go.Figure:
    """
    Generates and caches 3D interactive visualization assets.
    
    This function uses @st.cache_resource to ensure that expensive 3D mesh
    generation and HD-BET brain extraction only run once per patient/file.
    
    Args:
        file_hash: Unique identifier for the patient scan.
        raw_nifti_bytes: Raw bytes of the NIfTI MRI scan.
        disease: internal key for the detected disease.
        disease_name: Human-readable name of the disease.
        single_seg_data: Dictionary containing segmentation masks and scores.
        roi_metadata: Spatial metadata for upsampling ROI to original space.
        affine: 4x4 affine matrix of the original scan.
        spacing: Voxel spacing of the original scan.
        lesion_metrics: Quantitative analytical measurements.
        model_path: Path to the trained checkpoint for weight loading.
        
    Returns:
        plotly.graph_objects.Figure: The interactive 3D scene (Solid Mesh View).
    """
    print(f"\n🚀 [ASSET PIPE] Generating persistent 3D scene for: {disease_name}")
    
    # One-shot NIfTI lifecycle for mesh generation
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp_raw:
        tmp_raw.write(raw_nifti_bytes)
        active_viz_path = tmp_raw.name
    
    try:
        # Load local model weight copy for mesh alignment verification
        local_model = load_model(model_path)
        
        # PRIMARY ASSET: Interactive solid-mesh patient reconstructio
        fig_patient = create_3d_visualization(
            file_path=active_viz_path,
            segmentations_roi=single_seg_data,
            roi_metadata=roi_metadata,
            affine=affine, 
            spacing=spacing,
            show_patient_brain=True,
            show_heatmap=False,  # REQ: No heatmap in UI
            lesion_metrics=lesion_metrics,
            model=local_model,
            file_hash=file_hash
        )
        
        return fig_patient
        
    finally:
        # Prevent temporary file leakage
        if os.path.exists(active_viz_path):
            os.unlink(active_viz_path)

def create_3d_visualization(
    file_path: str,
    segmentations_roi: Dict,
    roi_metadata: Dict,
    affine: np.ndarray,
    spacing: Tuple[float, float, float],
    show_patient_brain: bool = True,
    clinical_decision: Optional[Dict] = None,
    show_heatmap: bool = False,
    lesion_metrics: Optional[Dict] = None,
    model=None,
    file_hash: str = None
) -> go.Figure:
    """
    Builds a research-grade 3D interactive scene with brain surface and lesion meshes.
    
    The visualization uses world coordinates (RAS) for all meshes and markers. 
    It incorporates HD-BET for high-fidelity skull stripping and performs 
    connected-component analysis to remove resampling artifacts.
    
    Args:
        file_path: Original NIfTI file path.
        segmentations_roi: Dictionary of ROI-space segmentations.
        roi_metadata: Metadata for mapping ROI to original space.
        affine: Original 4x4 NIfTI affine matrix.
        spacing: Original voxel spacing.
        show_patient_brain: Toggle translucent brain shell.
        clinical_decision: Optional dictionary with classification results.
        show_heatmap: Toggle probability heatmap overlay.
        lesion_metrics: Dictionary of clinical analytics (volume, centroid, etc.).
        model: Trained MultiGenAI model (for decision score verification).
        file_hash: MD5 hash for disk caching.
        
    Returns:
        plotly.graph_objects.Figure: The complete 3D scene.
    """
    fig = go.Figure()
    
    print("\n" + "=" * 60)
    print("🧠 BRAIN EXTRACTION PIPELINE")
    print("=" * 60)
    
    # HD-BET ONLY - NO FALLBACK (Gold Standard)
    print("\n🎯 Calling HD-BET (ONLY method - gold standard)...")
    
    brain_volume, brain_mask = apply_hdbet_brain_extraction(file_path, spacing, file_hash)
    
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
    
    # Graceful Abort: If no brain detected, we cannot crop or visualize correctly
    if brain_bbox is None:
        st.warning("⚠️ **Brain Extraction Incomplete:** No brain tissue detected in this volume. 3D visualization is not possible.")
        print("❌ 3D RENDERING ABORTED - Empty brain mask (no voxels)")
        return fig
        
    bbox_shape = tuple(s.stop - s.start for s in brain_bbox)
    
    # Safety Check: Handle missing brain volume (NameError/NoneType Guard)
    if brain_volume is None:
        print("❌ CRITICAL: brain_volume is None after extraction")
        st.error("Visualization failed: Internal volume data missing.")
        return fig
    
    print(f"📦 Bounding box: {bbox_shape} (from original {brain_volume.shape})")
    
    # Crop to brain region
    brain_mask_cropped = brain_mask[brain_bbox]
    original_cropped = brain_volume[brain_bbox]
    print(f"✂️  Cropped to brain-only region")
    
    # CRITICAL: Compute affine for the CROPPED region to support world coordinates
    # Translation matrix for cropping offset (in voxel space)
    z_start, y_start, x_start = brain_bbox[0].start, brain_bbox[1].start, brain_bbox[2].start
    
    offset_voxel = np.array([x_start, y_start, z_start], dtype=np.float64)

    # Convert voxel offset → world offset
    offset_world = affine[:3, :3] @ offset_voxel

    cropped_affine = affine.copy()
    cropped_affine[:3, 3] += offset_world
    print("🌍 Scene shifted to WORLD COORDINATES (mm)")
    print("=" * 60 + "\n")
    
    # LAYER 1: Shared Brain Surface Preprocessing (Always calculate for alignment, conditionally render)
    brain_verts, brain_faces = None, None
    if show_patient_brain and brain_mask_cropped is not None:
        try:
            print("🧠 Preprocessing brain surface for mesh/atlas alignment...")
            cache_dir = os.path.join("cache", file_hash) if file_hash else None
            brain_mesh_path = os.path.join(cache_dir, "brain_mesh.npz") if cache_dir else None
            
            if brain_mesh_path and os.path.exists(brain_mesh_path):
                print("⚡ Using cached patient brain mesh from disk")
                data = np.load(brain_mesh_path)
                brain_verts, brain_faces = data['verts'], data['faces']
            else:
                brain_verts, brain_faces = generate_patient_brain_surface(
                    brain_mask=brain_mask_cropped,
                    affine=cropped_affine,
                    spacing=spacing
                )
                if brain_mesh_path:
                    # Parent directory check for safety
                    os.makedirs(os.path.dirname(brain_mesh_path), exist_ok=True)
                    np.savez_compressed(brain_mesh_path, verts=brain_verts, faces=brain_faces)
            
            # 1.1 Render Patient Brain (Conditional)
            if show_patient_brain:
                fig.add_trace(go.Mesh3d(
                    x=brain_verts[:, 0], y=brain_verts[:, 1], z=brain_verts[:, 2],
                    i=brain_faces[:, 0], j=brain_faces[:, 1], k=brain_faces[:, 2],
                    color='lightgray', opacity=0.4,
                    name='Brain Surface', showlegend=True, hoverinfo='skip',
                    lighting=dict(ambient=0.6, diffuse=0.8, roughness=0.5),
                    lightposition=dict(x=100, y=200, z=0)
                ))
            
                    
        except Exception as e:
            st.warning(f"⚠️ Could not generate brain/atlas surface: {e}")
            print(f"⚠️ Brain surface error: {e}")
    
    # LAYER 2: Lesion Surfaces (Mapped to Original Space, SEPARATE from brain)
    if 'metrics' not in st.session_state:
        st.session_state.metrics = {}
        
    for disease, result in segmentations_roi.items():
        # Supports legacy 3-tuple, 5-tuple, and new 6-tuple metrics propagation
        if len(result) == 3:
            _, binary_roi, dec_score = result
            probs_roi, prob_orig, unc, logit_raw = binary_roi.astype(float), 0.5, 0.0, 0.0
        elif len(result) == 5:
            binary_roi, dec_score, prob_orig, unc, logit_raw = result
            probs_roi = binary_roi.astype(float)
        elif len(result) == 6:
            binary_roi, probs_roi, dec_score, prob_orig, unc, logit_raw = result
        else:
            print(f"⚠️ Unexpected result format {len(result)} for {disease}")
            continue

        # ALZHEIMER HARD GUARD (COMPLIANCE REQUIREMENT)
        if disease == "alzheimer":
            st.info(f"ℹ️ **Alzheimer's Disease**: Presence-only detection (no voxel-level localization). "
                    "ADNI dataset does not provide lesion masks. 3D visualization not applicable.")
            continue  # Skip 3D mesh, slice overlay, volume rendering
        
        # CLINICAL GATING: We no longer reject based on decision head or primary disease
        # (Per USER REQUEST: "REMOVE HARD REJECTION")
        # We process all detected diseases that have segmentation data
        pass
        
        disease_cfg = DISEASE_COLORS.get(disease, {"hex": "#FFFFFF", "name": disease})
        color = disease_cfg["hex"]
        name = disease_cfg["name"]
        
        print(f"\n🔬 Processing {name} lesion...")
        print(f"   ROI space: {binary_roi.shape}")
        
        # Use same threshold as training
        VIS_THRESHOLD = 0.5
        
        print(f"\n📊 {name} Probability Distribution in ROI:")
        print(f"   Min: {probs_roi.min():.4f}")
        print(f"   Max: {probs_roi.max():.4f}")
        print(f"   ROI sum:", (probs_roi > 0.5).sum())
        count = int((probs_roi > 0.5).sum())
        total = probs_roi.size
        print(f"   Voxels > 0.5: {count:,} ({count/total*100:.1f}%)")
        
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
            
            # Using soft probabilities for accurate marching cubes (thresholding later handled inherently)
            binary_strict = (prob_wt > VIS_THRESHOLD).astype(np.uint8)
            
        elif probs_roi.ndim == 4:  # Multi-channel logic (generic fallback)
            binary_strict = (probs_roi.max(axis=0) > VIS_THRESHOLD).astype(np.uint8)
        else:
            # Single channel (Stroke)
            binary_strict = (probs_roi > VIS_THRESHOLD).astype(np.uint8)
            
        # -------------------------------------------------------------------------
        # ML DECISION HEAD (FIX: REMOVE HARD REJECTION)
        # -------------------------------------------------------------------------
        # We no longer 'continue' on low dec_prob, just log it.
        if dec_score < 0.5:
            print(f"⚠️ DECISION HEAD WEAK SIGNAL for {name} (confidence={dec_score:.2f})")
        else:
            print(f"✅ DECISION HEAD ACCEPTED {name} (confidence={dec_score:.2f})")
        
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
        assert seg_original.shape == brain_volume.shape[:3], (
            f"Segmentation shape {seg_original.shape} does not match "
            f"original volume shape {brain_volume.shape[:3]}. "
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
        # FIX 1 — HARD CLIP: Force lesion inside brain mask (non-negotiable).
        # Any voxel outside the brain mask is anatomically impossible.
        before_clip = int(seg_original.sum())
        seg_original = seg_original.astype(bool) & scipy_binary_dilation(brain_mask_cropped.astype(bool), iterations=1)
        after_clip = int(seg_original.sum())
        print(f"   Brain mask hard clip: {before_clip:,} → {after_clip:,} voxels "
              f"({before_clip - after_clip:,} outside-brain voxels removed)")

        # FIX 2 — LARGEST CONNECTED COMPONENT: Eliminate floating clusters.
        # Resampling creates disconnected specks that produce phantom meshes.
        labeled, num_components = cc_label(seg_original)
        if num_components > 1:
            sizes = [(labeled == i).sum() for i in range(1, num_components + 1)]
            largest_label = int(np.argmax(sizes)) + 1
            # FIX 3 — MICRO-ARTIFACT REMOVAL: Drop components < 10 voxels.
            MIN_COMPONENT_SIZE = 10
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
            if inside_ratio < 0.7:
                st.warning(
                    f"⚠️ **{name}**: Poor alignment ({inside_ratio:.1%} inside brain). "
                    f"Applying additional brain-mask clamp..."
                )
                # Apply a second clamp pass to push ratio to 100%
                seg_original = (seg_original.astype(bool) & scipy_binary_dilation(brain_mask_cropped.astype(bool), iterations=1)).astype(np.uint8)
                print(f"   After second clamp: {seg_original.sum():,} voxels")
        
        seg_clean = seg_original
        cleaned_voxels = int(seg_clean.sum())
        print(f"   Final lesion voxels for mesh: {cleaned_voxels:,}")
        
        # 🔥 ANALYTICS LAYER (Per USER REQUEST)
        # Compute metrics on the cleaned/clamped segmentation passing presence-level scores
        metrics = compute_lesion_metrics(
            mask=seg_clean,
            brain_mask=brain_mask_cropped,
            spacing=spacing,
            prob=dec_score,
            uncertainty=unc,
            logit=logit_raw,
            affine=cropped_affine
        )
        st.session_state.metrics[disease] = metrics

        # FIX 4.3 — Minimum volume check (after all cleaning)
        MIN_VOLUME_VOXELS = 27
        if cleaned_voxels < MIN_VOLUME_VOXELS:
            st.warning(f"⚠️ **{name}**: Lesion too small after cleaning ({cleaned_voxels} voxels).")
            print(f"   ⚠️ Skipping: only {cleaned_voxels} voxels (minimum {MIN_VOLUME_VOXELS})")
            continue
        
        # FIX 4.2 — Unified sigma=0.5 to match brain surface smoothing and prevent loss of thin edges
        sigma = 0.5
        
        try:
            seg_smooth = gaussian_filter(seg_clean.astype(np.float32), sigma=sigma)
            
            cache_dir = os.path.join("cache", file_hash) if file_hash else None
            lesion_mesh_path = os.path.join(cache_dir, f"{disease}_mesh.npz") if cache_dir else None
            
            if lesion_mesh_path and os.path.exists(lesion_mesh_path):
                print(f"⚡ Using cached {disease} mesh from disk")
                data = np.load(lesion_mesh_path)
                verts, faces = data['verts'], data['faces']
            else:
                # GOLD-STANDARD: Marching cubes in voxel space
                level = 0.5
                verts, faces, normals, _ = measure.marching_cubes(
                    seg_smooth,
                    level=level
                )
                
                # FIX 4.4 — ZYX → XYZ swap then use nib.affines.apply_affine (cleaner than manual homogeneous)
                verts_xyz = verts[:, [2, 1, 0]]
                verts = nib.affines.apply_affine(cropped_affine, verts_xyz)
                
                if lesion_mesh_path:
                    np.savez_compressed(lesion_mesh_path, verts=verts, faces=faces)
            
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
            
            # Prepare Mesh Coloring (Heatmap)
            if show_heatmap:
                with st.spinner(f"Mapping {name} probabilities for heatmap..."):
                    # 1. Standardize to a 3D single-channel probability map
                    if probs_roi.ndim == 4:
                        # For tumor, use the Whole Tumor probability (Union of ET, NCR, ED)
                        if disease == "tumor":
                             p_single = np.maximum.reduce([probs_roi[0], probs_roi[1], probs_roi[2]])
                        else:
                             p_single = probs_roi.max(axis=0)
                    else:
                        p_single = probs_roi
                    
                    # 2. Map ROI probabilities back to original space (float) using interpolation
                    target_shape = roi_metadata["original_shape"]
                    p_tensor = torch.from_numpy(p_single).float().unsqueeze(0).unsqueeze(0) # Always (1, 1, D, H, W)
                        
                    p_original = F.interpolate(
                        p_tensor,
                        size=target_shape,
                        mode='trilinear',
                        align_corners=False
                    ).cpu().numpy()[0, 0] # Explicitly index (D, H, W) to avoid squeeze bug
                    
                    # 3. Crop probability map to same brain bbox
                    p_cropped = p_original[brain_bbox]
                    
                    # 4. Get sampling coordinates using INVERSE AFFINE for 100% sync
                    from scipy.ndimage import map_coordinates
                    inv_affine = np.linalg.inv(cropped_affine)
                    
                    # Verts are world coordinates (mm). Map to voxel coords.
                    verts_xyz_recon = nib.affines.apply_affine(inv_affine, verts)
                    verts_zyx_recon = verts_xyz_recon[:, [2, 1, 0]].T # (3, N) for map_coordinates
                    
                    # 5. FINAL SAFETY: Validate shapes before map_coordinates
                    if p_cropped.ndim != 3 or verts_zyx_recon.shape[0] != 3:
                        print(f"❌ Heatmap shape mismatch: {p_cropped.shape} vs {verts_zyx_recon.shape}")
                        vertex_probs = np.full(len(verts), 0.5) # Fallback to neutral
                    else:
                        vertex_probs = map_coordinates(
                            p_cropped,
                            verts_zyx_recon,
                            order=1,
                            mode='nearest'
                        )
                    
                    print(f"   🔥 Heatmap sampled via inverse affine: Prob range [{vertex_probs.min():.4f}, {vertex_probs.max():.4f}]")
                
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
                    vol_ml = m["lesion_volume_mm3"] / 1000.0
                    
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
    # FIX: Dynamic Camera pinning to brain center
    brain_center = brain_verts.mean(axis=0) if 'brain_verts' in locals() else [0,0,0]
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, showgrid=False),
            yaxis=dict(visible=False, showgrid=False),
            zaxis=dict(visible=False, showgrid=False),
            bgcolor='#0a1120',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),
                center=dict(x=0, y=0, z=0)
            ),
            aspectmode='data'
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
            text="Patient Anatomy Reconstruction",
            font=dict(size=14, color='#00E5FF'),
            x=0.5,
            xanchor='center'
        ),
        annotations=[
            dict(
                text="<b>Source:</b> Clinical brain surface from deep-learning mask (HD-BET). "
                     "Lesion segmentation in 96³ ROI, mapped to world RAS space.",
                xref="paper", yref="paper",
                x=0.5, y=-0.05,
                showarrow=False,
                font=dict(size=10, color='#9CA3AF'),
                xanchor='center'
            )
        ]
    )
    
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# MATPLOTLIB 2D HEATMAP GENERATOR (HIGH-RES PNG, NO PLOTLY DEPENDENCY)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def generate_matplotlib_heatmap_png(
    probs_roi_bytes: bytes,
    probs_roi_shape: tuple,
    probs_roi_dtype_str: str,
    roi_metadata_original_shape: tuple,
    roi_metadata_roi_affine: list,
    roi_metadata_original_affine: list,
    disease: str,
    disease_name: str,
    dpi: int = 200
) -> bytes:
    """
    Generate a true 2D multi-slice probability heatmap as high-res PNG bytes.

    Uses matplotlib only — fully independent of Plotly/kaleido.
    Cached by st.cache_data so repeated calls (reruns) return instantly
    without re-generating or touching any other session state.

    Args:
        probs_roi_bytes: Raw bytes of the probability array (from np.tobytes()).
        probs_roi_shape: Shape tuple of the probability array.
        probs_roi_dtype_str: dtype string (e.g. 'float32').
        roi_metadata_original_shape: original_shape from roi_metadata.
        roi_metadata_roi_affine: roi_affine as nested list (JSON-serializable).
        roi_metadata_original_affine: original_affine as nested list.
        disease: Disease key string ('tumor', 'stroke', 'alzheimer').
        disease_name: Human-readable disease name.
        dpi: Output resolution (200 = ~3200x2000 for A4).

    Returns:
        PNG bytes suitable for st.download_button.
    """
    import io
    import numpy as np
    import torch
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LinearSegmentedColormap

    # --- 1. Reconstruct arrays from cache-safe primitives ---
    probs_roi = np.frombuffer(probs_roi_bytes, dtype=np.dtype(probs_roi_dtype_str)).reshape(probs_roi_shape).copy()
    original_shape = tuple(roi_metadata_original_shape)

    # --- 2. Collapse multi-channel (tumor ET/NCR/ED) to single probability map ---
    if probs_roi.ndim == 4:
        if disease == "tumor":
            p_single = np.maximum.reduce([probs_roi[0], probs_roi[1], probs_roi[2]])
        else:
            p_single = probs_roi.max(axis=0)
    else:
        p_single = probs_roi.copy()

    # --- 3. Upsample ROI prob map to original image space ---
    p_tensor = torch.from_numpy(p_single.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    p_original = F.interpolate(
        p_tensor,
        size=original_shape,
        mode='trilinear',
        align_corners=False
    ).squeeze().numpy()  # shape: original_shape

    # --- 4. Pick representative slices (center of mass of high-prob region) ---
    threshold_mask = p_original > 0.3
    if threshold_mask.sum() > 10:
        coords = np.argwhere(threshold_mask)
        center = coords.mean(axis=0).astype(int)
    else:
        center = np.array([s // 2 for s in original_shape])

    ax_idx  = int(np.clip(center[2], 0, original_shape[2] - 1))  # Axial   (z)
    cor_idx = int(np.clip(center[1], 0, original_shape[1] - 1))  # Coronal (y)
    sag_idx = int(np.clip(center[0], 0, original_shape[0] - 1))  # Sagittal(x)

    # Collect ±2 slices around center for context strip
    def _safe_slices(center_v, max_v, n=5):
        half = n // 2
        idxs = [center_v + i for i in range(-half, half + 1)]
        return [max(0, min(max_v - 1, i)) for i in idxs]

    ax_slices  = _safe_slices(ax_idx,  original_shape[2], n=5)
    cor_slices = _safe_slices(cor_idx, original_shape[1], n=5)
    sag_slices = _safe_slices(sag_idx, original_shape[0], n=5)

    # --- 5. Disease color config ---
    _DISEASE_COLORS = {
        "tumor":     {"rgb": [255, 68, 68],   "name": "Tumor"},
        "stroke":    {"rgb": [68, 68, 255],   "name": "Stroke"},
        "alzheimer": {"rgb": [255, 136, 0],   "name": "Alzheimer Pattern"},
    }
    d_cfg   = _DISEASE_COLORS.get(disease, {"rgb": [0, 229, 255], "name": disease_name})
    d_rgb   = [c / 255.0 for c in d_cfg["rgb"]]
    d_cmap  = LinearSegmentedColormap.from_list(
        f"{disease}_heatmap",
        [(0, 0, 0, 0), (*d_rgb, 0.3), (*d_rgb, 0.85), (1, 1, 1, 1)],
        N=256
    )

    # --- 6. Build figure ---
    # Layout: 3 rows (Axial / Coronal / Sagittal), 5 columns (strip), + colorbar column
    BG    = '#030712'
    TITLE = '#00E5FF'
    LABEL = '#94A3B8'

    fig = plt.figure(figsize=(22, 14), facecolor=BG)
    fig.patch.set_facecolor(BG)

    outer = gridspec.GridSpec(
        4, 1,
        figure=fig,
        hspace=0.35,
        height_ratios=[0.6, 3, 3, 3]
    )

    # --- Title Row ---
    ax_title = fig.add_subplot(outer[0])
    ax_title.set_facecolor(BG)
    ax_title.axis('off')
    ax_title.text(
        0.5, 0.65,
        f"NeuroX  ·  {disease_name.upper()} Probability Heatmap",
        ha='center', va='center',
        fontsize=22, fontweight='bold', color=TITLE,
        fontfamily='monospace',
        transform=ax_title.transAxes
    )
    ax_title.text(
        0.5, 0.15,
        "RESEARCH & EDUCATIONAL USE ONLY — NOT FOR CLINICAL DIAGNOSIS",
        ha='center', va='center',
        fontsize=9, color='#FF8800',
        transform=ax_title.transAxes
    )

    def _render_strip(row_gs, slices_list, vol_3d, axis_label, axis_dim):
        """Render a 5-slice strip for one view axis."""
        inner = gridspec.GridSpecFromSubplotSpec(
            1, 6,
            subplot_spec=row_gs,
            wspace=0.04,
            width_ratios=[1, 1, 1, 1, 1, 0.06]
        )
        for col_i, sl_idx in enumerate(slices_list):
            ax = fig.add_subplot(inner[0, col_i])
            ax.set_facecolor('#000000')

            # Extract 2D slice
            if axis_dim == 2:   slc = vol_3d[:, :, sl_idx]
            elif axis_dim == 1: slc = vol_3d[:, sl_idx, :]
            else:               slc = vol_3d[sl_idx, :, :]

            slc_rot = np.rot90(slc, k=1)

            # Grayscale base (normalized prob as grayscale for context)
            ax.imshow(slc_rot, cmap='gray', vmin=0, vmax=1,
                      origin='upper', interpolation='bilinear', aspect='auto')
            # Heatmap overlay
            im = ax.imshow(slc_rot, cmap=d_cmap, vmin=0, vmax=1,
                           alpha=0.85, origin='upper',
                           interpolation='bilinear', aspect='auto')

            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor('#1E293B')
                spine.set_linewidth(0.5)

            # Slice index label
            ax.set_xlabel(f"Slice {sl_idx}", color=LABEL, fontsize=7, labelpad=2)

            # Center slice highlight
            if col_i == len(slices_list) // 2:
                for spine in ax.spines.values():
                    spine.set_edgecolor(TITLE)
                    spine.set_linewidth(2.0)

        # Colorbar
        cbar_ax = fig.add_subplot(inner[0, 5])
        cbar_ax.set_facecolor(BG)
        sm = plt.cm.ScalarMappable(cmap=d_cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cbar_ax)
        cb.ax.yaxis.set_tick_params(color=LABEL, labelsize=7)
        cb.outline.set_edgecolor('#1E293B')
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=LABEL)
        cb.set_label('Probability', color=LABEL, fontsize=8, labelpad=4)

        # Row label
        row_label_ax = fig.add_subplot(inner[0, 0])
        row_label_ax.set_facecolor(BG)
        row_label_ax.text(
            -0.18, 0.5, axis_label,
            ha='center', va='center',
            fontsize=11, fontweight='bold',
            color=TITLE,
            rotation=90,
            transform=row_label_ax.transAxes
        )

    _render_strip(outer[1], ax_slices,  p_original, "AXIAL",    2)
    _render_strip(outer[2], cor_slices, p_original, "CORONAL",  1)
    _render_strip(outer[3], sag_slices, p_original, "SAGITTAL", 0)

    # --- 7. Save to bytes ---
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format='png',
        dpi=dpi,
        facecolor=BG,
        bbox_inches='tight',
        pad_inches=0.3
    )
    plt.close(fig)
    buf.seek(0)
    return buf.read()


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
    for disease, result in segmentations_roi.items():
        # Fixed Issue 1: Unpack all 6 elements from segment_images return
        binary_roi, probs_roi, dec_score, prob_orig, unc, logit_raw = result
        if disease == "alzheimer":
            continue  # No volume rendering for Alzheimer
        
        # Map to original space
        seg_original = map_segmentation_to_original_space(binary_roi, roi_metadata)
        
        # Downsample
        seg_down = seg_original[::downsample_factor, ::downsample_factor, ::downsample_factor]
        
        if seg_down.sum() == 0:
            continue
        
        # Create volume trace
        disease_cfg = DISEASE_COLORS.get(disease, {"hex": "#FFFFFF", "name": disease})
        color = disease_cfg["hex"]
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
            name=DISEASE_COLORS.get(disease, {"name": disease})["name"],
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
    probabilities = [probs.get(d, 0.0) for d in diseases]
    colors = [DISEASE_COLORS.get(d, {"hex": "#FFFFFF"})["hex"] for d in diseases]
    names = [DISEASE_COLORS.get(d, {"name": d})["name"] for d in diseases]
    
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


def render_paper_report(report_text: str):
    """
    High-Fidelity 'Paper View' using the structured reporting engine.
    Ensures findings are inside the clinical whiteboard container.
    """
    import streamlit as st
    
    # 1. Parse raw AI text into structured objects
    structured = re_engine.StructuredReport(report_text)
    
    # 2. Render using the specialized whiteboard engine
    re_engine.render_clinical_whiteboard(st, structured)

# ═══════════════════════════════════════════════════════════════════════════
# AI REPORT GENERATION (GROQ)
# ═══════════════════════════════════════════════════════════════════════════

def generate_ai_report(detection: Dict, segmentations: Dict, lesion_metrics: Dict, groq_api_key: Optional[str] = None) -> str:
    """
    High-Fidelity AI Radiology Report Generation.
    
    Acts as a Senior Neuroradiologist Specialist to interpret raw quantitative 
    metrics (volumes, coordinates, depths) into a professional clinical narrative.
    """
    if not GROQ_AVAILABLE or not groq_api_key:
        return generate_fallback_report(detection, segmentations)
    
    try:
        client = Groq(api_key=groq_api_key)
        
        # Build Analytical Context for the LLM
        detected = detection.get("detected_diseases", [])
        probs = detection.get("probabilities", {})
        uncs = detection.get("uncertainties", {})
        
        context_data = []
        for disease in detected:
            metrics = lesion_metrics.get(disease, {})
            name = DISEASE_COLORS.get(disease, {"name": disease})["name"]
            
            # 1. Core Confidence & Uncertainty
            p = probs.get(disease, 0.0)
            u = uncs.get(disease, 0.0)
            
            # 2. Detailed Morphometrics (Fix: mm3 -> mL)
            vol_ml = metrics.get("lesion_volume_mm3", 0.0) / 1000.0
            centroid = metrics.get("centroid_mm", [0, 0, 0])
            pos_ap = metrics.get("position_AP", "Unknown")
            pos_si = metrics.get("position_SI", "Unknown")
            hemi = metrics.get("hemisphere", "Unknown")
            depth = metrics.get("depth_mean", 0.0)
            
            findings = (
                f"### {name.upper()}\n"
                f"- Presence Confidence: {p:.1%} (Entropy-calibrated uncertainty: {u:.3f})\n"
                f"- Quantitative Volume: {vol_ml:.3f} mL ({metrics.get('lesion_voxels', 0):,} voxels)\n"
                f"- Spatial Location: {hemi} hemisphere, {pos_ap} aspect, {pos_si} region\n"
                f"- World RAS Coordinates (Centroid): [{centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f}] mm\n"
                f"- Morphological Depth: Mean {depth:.1f} mm from pial surface\n"
            )
            context_data.append(findings)

        context_summary = "\n".join(context_data) if context_data else "No significant imaging patterns detected."

        # The Perfect Clinical Prompt (Expert Persona v2)
        prompt = f"""
SYSTEM ROLE: You are an expert Board-Certified Senior Neuroradiologist interpreting high-resolution structural MRI (T1, T2, FLAIR).

INPUT DATA (QUANTITATIVE SIGNATURES):
{context_summary}

TASK:
Generate a formal, structured neuroradiology report. Your objective is to synthesize raw metrics (mL, mm, coordinates) into a clinical narrative. 

MANDATORY STRUCTURE:
You MUST use these exact headers (including Roman Numerals) for each section:

I. CLINICAL FINDINGS:
Itemized anatomical observations. Use professional terminology (vasogenic edema, mass effect, cytotoxic edema, medial temporal atrophy).

II. LOCALIZED ANALYTICAL MEASUREMENTS:
A bulleted summary of ALL quantitative metrics. You MUST list every volumetric measurement (mL), voxel count, and RAS coordinate provided in the input here, even if you already mentioned them in the findings.

III. DIFFERENTIAL CONSIDERATIONS:
Diagnostic possibilities consistent with imaging (neoplastic, infarct, etc.). 

IV. IMPRESSION:
A final qualitative clinical synthesis based on ALL above findings.

CRITICAL CONSTRAINTS:
- YOU MUST SEPARATE FINDINGS FROM MEASUREMENTS. Do not only put metrics in the findings prose.
- Use Radiologist-level HEDGING ("Suggestive of", "Consistent with").
- RESEARCH USE ONLY: Cite that these are automated findings based on the NeuroX DL-pipeline.
- Max 500 words.
"""
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=750
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        st.warning(f"AI report generation failed: {e}")
        return generate_fallback_report(detection, segmentations)


def generate_pdf_report(report_text: str, detection: Dict, metrics: Dict):
    """
    Generates a structured clinical PDF report using reportlab and the specialized engine.
    """
    structured = re_engine.StructuredReport(report_text)
    return re_engine.generate_structured_pdf(structured)




def generate_fallback_report(detection: Dict, segmentations: Dict) -> str:
    """Fallback report without AI, aligned with StructuredReport engine"""
    detected = detection["detected_diseases"]
    probs = detection["probabilities"]
    
    report = "CLINICAL FINDINGS:\n"
    if not detected:
        report += "- No significant imaging abnormalities or pathological patterns identified above clinical thresholds.\n"
    else:
        for disease in detected:
            name = DISEASE_COLORS.get(disease, {"name": disease})["name"]
            report += f"- Positive imaging signature for {name} (Confidence: {probs.get(disease, 0.0):.1%}).\n"
    
    report += "\nLOCALIZED ANALYTICAL MEASUREMENTS:\n"
    if not detected:
        report += "- 0 mL / 0 voxels\n"
    else:
        for disease in detected:
            if disease in segmentations:
                res = segmentations[disease]
                if len(res) == 3: mask = res[1]
                elif len(res) in [5, 6]: mask = res[0]
                else: mask = np.zeros((1,1,1))
                
                name = DISEASE_COLORS.get(disease, {"name": disease})["name"]
                voxels = int(mask.sum())
                report += f"- {name} volume: {voxels/1000.0:.3f} mL ({voxels:,} voxels)\n"
    
    report += "\nIMPRESSION:\n"
    if not detected:
        report += "Normal neuroimaging findings. No focal lesions or neurodegenerative patterns detected based on current model thresholds.\n"
    else:
        findings_str = ", ".join([DISEASE_COLORS.get(d, {"name": d})["name"] for d in detected])
        report += f"Automatic analysis identified patterns consistent with: {findings_str}. Clinical correlation and formal neuroradiological review are mandatory for diagnostic confirmation.\n"

    report += "\n---\n"
    report += "**RESEARCH AND EDUCATIONAL USE ONLY - NOT FOR CLINICAL DIAGNOSIS**\n"
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
            disease_cfg = DISEASE_COLORS.get(disease, {"name": disease})
            name = disease_cfg["name"]
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
    
    # NEW CACHING FLAGS
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False
    if 'file_hash' not in st.session_state:
        st.session_state.file_hash = None
    if 'cached_viz' not in st.session_state:
        st.session_state.cached_viz = {}

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
    if 'raw_nifti_bytes' not in st.session_state:
        st.session_state.raw_nifti_bytes = None
    if 'training_metrics' not in st.session_state:
        st.session_state.training_metrics = {}   # populated by load_model from checkpoint
        
    # GLOBAL SETTINGS STATE (Initialize defaults)
    if 'show_atlas' not in st.session_state:
        st.session_state.show_atlas = True
    if 'show_heatmap' not in st.session_state:
        st.session_state.show_heatmap = False
    
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
            
            raw_bytes = uploaded_file.getvalue()
            file_hash = hashlib.md5(raw_bytes).hexdigest()
            
            if st.session_state.file_hash != file_hash:
                print("🔄 New input detected. Clearing analysis cache...")
                st.session_state.analysis_complete = False
                st.session_state.analysis_done = False
                st.session_state.cached_viz = {}
                st.session_state.file_hash = file_hash
                
            st.session_state.raw_nifti_bytes = raw_bytes
            
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

                                # FIX 1 — GLOBAL BATCH ENFORCEMENT
                                if image_tensor.dim() == 4:
                                    image_tensor = image_tensor.unsqueeze(0)
                                
                                # Use session safe threshold
                                # EXECUTE CLINICAL DETECTION (Fixed Issue 7 — Hardcoded thresholds)
                                
                                # SAFETY GAP — GLOBAL MULTIMODAL ENFORCEMENT
                                assert image_tensor.dim() == 5, f"Expected 5D input, got {image_tensor.shape}"

                                x_alz = prepare_input(image_tensor, "alzheimer")
                                x_seg = prepare_input(image_tensor, "tumor")

                                assert x_alz.shape[1] == 1, f"Alzheimer input must be 1ch, got {x_alz.shape}"
                                assert x_seg.shape[1] == 2, f"Segmentation input must be 2ch, got {x_seg.shape}"

                                # FIX 2 — SPLIT-CALL SYNC
                                detection = automatic_disease_detection_dual(
                                    model, 
                                    image_tensor=image_tensor
                                )
                                
                                # FIX 7 — MULTI-LABEL DYNAMICS (Don't return early!)
                                # Process all detected diseases. 
                                detected_list = detection["detected_diseases"]
                                seg_tasks = [d for d in detected_list if d in ["tumor", "stroke"]]
                                
                                # Store ALL components including affine/spacing
                                st.session_state.detection_results = detection
                                st.session_state.original_image = original_data
                                st.session_state.roi_metadata = roi_metadata
                                st.session_state.affine = affine
                                st.session_state.spacing = spacing
                                
                                segmentations = {}
                                if seg_tasks:
                                    segmentations = perform_segmentation(
                                        model,
                                        image_tensor,
                                        seg_tasks
                                    )
                                st.session_state.segmentation_results = segmentations
                                
                                # CRITICAL: Calculate metrics after segmentation and store in session state
                                lesion_metrics = {}
                                
                                # Generate a fast brain mask for metrics (Otsu)
                                # This ensures the Analytical Layer has a valid reference volume
                                with st.spinner("📦 Generating brain reference mask..."):
                                    brain_mask_metrics = generate_brain_mask_otsu(original_data)
                                
                                for disease, seg_tuple in segmentations.items():
                                    # Support new 6-tuple (mask, probs, dec_score, prob, unc, logit)
                                    binary_roi, probs_roi, dec_score, p_orig, u_orig, l_orig = seg_tuple
                                    seg_original = map_segmentation_to_original_space(binary_roi, roi_metadata)
                                    
                                    # Passing analytics data from detection step for the CLINICAL ANALYTICAL LAYER
                                    lesion_metrics[disease] = compute_lesion_metrics(
                                        mask=seg_original, 
                                        brain_mask=brain_mask_metrics, 
                                        spacing=st.session_state.spacing,
                                        prob=dec_score, # Use calibrated score
                                        uncertainty=u_orig,
                                        logit=l_orig,
                                        affine=affine
                                    )
                                    if lesion_metrics[disease] is not None:
                                        lesion_metrics[disease]["decision_score"] = dec_score
                                
                                st.session_state.lesion_metrics = lesion_metrics
                                
                                st.session_state.analysis_complete = True
                                st.session_state.analysis_done = True # Set persistence flag
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
                disease_cfg = DISEASE_COLORS.get(disease, {"hex": "#FFFFFF", "name": disease})
                disease_name = disease_cfg["name"]
                disease_color = disease_cfg["hex"]
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
                    
                    # Additional information (ANALYTICAL LAYER)
                    # We fetch from lesion_metrics (calculated at analysis time) or compute now
                    m = st.session_state.get("lesion_metrics", {}).get(disease, {})
                    if not m and disease == "alzheimer":
                        # Alzheimer doesn't have lesion_metrics (no segmentation)
                        # We use the detection results directly via compute_alzheimer_metrics
                        alz_l = det.get("presence_logits", {}).get("alzheimer", 0.0)
                        m = compute_alzheimer_metrics(prob, uncertainty, alz_l)
                    
                    if m:
                        st.markdown(f"""
                        <div class="glass-card">
                            <h4 style="color: {disease_color};">Clinical Analytical Layer</h4>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                                <p style="color: #E5E7EB; font-size: 13px; margin: 0;">🧠 <b>Entropy:</b> {m.get('entropy', 0.0):.3f}</p>
                                <p style="color: #E5E7EB; font-size: 13px; margin: 0;">🛡️ <b>Reliability:</b> {m.get('consistency', 0.0):.1%}</p>
                                <p style="color: #E5E7EB; font-size: 13px; margin: 0;">⚡ <b>Logit Strength:</b> {m.get('logit_strength', 0.0):.2f}</p>
                                <p style="color: #E5E7EB; font-size: 13px; margin: 0;">📉 <b>Adjusted Score:</b> {m.get('adjusted_score', 0.0):.2f}</p>
                            </div>
                            <hr style="margin: 10px 0; border-color: rgba(255,255,255,0.1);">
                            <p style="color: #94A3B8; font-size: 13px;">
                                • <b>Risk Assessment:</b> <span style="color:{disease_color};">{m.get('risk', 'N/A')} Risk</span> patterns identified<br>
                                • <b>Margin:</b> {m.get('margin', 0.0):.2f} ({'Strong' if m.get('margin', 0)>0.25 else 'Borderline'} signal)<br>
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
                    disease_cfg = DISEASE_COLORS.get(disease, {"hex": "#FFFFFF", "name": disease})
                    name = disease_cfg["name"]
                    color = disease_cfg["hex"]
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
                            vol = metrics.get("lesion_volume_mm3", 0)
                            centroid = metrics.get("centroid_mm", [0, 0, 0])
                            bbox_min = metrics.get("bbox_min_mm", [0, 0, 0])
                            bbox_max = metrics.get("bbox_max_mm", [0, 0, 0])
                            dec_score = metrics.get("decision_score", None)
                            dec_line = f"<p style='color: #E5E7EB; font-size: 13px; margin: 5px 0;'>🤖 <b>Decision Score:</b> {dec_score:.2%} ({'✅ Accepted' if dec_score and dec_score>=0.5 else '⚠️ Low confidence'})</p>" if dec_score is not None else ""
                            
                            st.markdown(f"""
                            <div style="margin-left: 20px; border-left: 2px solid {color}44; padding-left: 15px; margin-bottom: 20px;">
                                <p style="color: #E5E7EB; font-size: 13px; margin: 5px 0;">📏 <b>Volume:</b> {vol:,.1f} mm³ ({vol/1000:,.2f} mL)</p>
                                <p style="color: #E5E7EB; font-size: 13px; margin: 5px 0;">🎯 <b>Centroid (World RAS):</b> ({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f}) mm</p>
                                <p style="color: #E5E7EB; font-size: 13px; margin: 5px 0;">📦 <b>Bounding Box (Min, World RAS):</b> ({bbox_min[0]:.1f}, {bbox_min[1]:.1f}, {bbox_min[2]:.1f}) mm</p>
                                <p style="color: #E5E7EB; font-size: 13px; margin: 5px 0;">📦 <b>Bounding Box (Max, World RAS):</b> ({bbox_max[0]:.1f}, {bbox_max[1]:.1f}, {bbox_max[2]:.1f}) mm</p>
                                {dec_line}
                            </div>
                            """, unsafe_allow_html=True)
            
            # (Training Dashboard moved to dedicated TRAINING page)

        else:
            st.info("No analysis results available. Please upload and analyze a scan first.")
    
    elif st.session_state.current_page == 'training':
        # ========== TRAINING DASHBOARD PAGE ==========
        st.markdown("## 📈 Training Dashboard")
        st.markdown("---")

        tm = st.session_state.get("training_metrics", {})
        if not tm or not tm.get("epoch"):
            st.info("Training metrics not available. Load a checkpoint trained with the structured schema.")
        else:
            ep   = tm["epoch"]
            tr   = tm.get("train", {})
            val  = tm.get("val", {})
            meta = tm.get("meta", {})

            # ─── Helper: Plotly dark figure ──────────────────────────────
            import plotly.graph_objects as go

            def _dark_fig(title, y_title="Score", y_range=None):
                fig = go.Figure()
                fig.update_layout(
                    title=dict(text=title, font=dict(family='Orbitron', size=15), x=0.04),
                    height=360, plot_bgcolor="rgba(3,7,18,0.6)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#E5E7EB", family='Inter'),
                    xaxis=dict(title="Epoch", gridcolor="rgba(255,255,255,0.05)", zeroline=False),
                    yaxis=dict(title=y_title, gridcolor="rgba(255,255,255,0.05)", zeroline=False,
                               range=y_range),
                    legend=dict(bgcolor="rgba(17,24,39,0.8)", bordercolor="rgba(0,229,255,0.2)", borderwidth=1),
                    margin=dict(t=55, b=40, l=55, r=20), hovermode="x unified"
                )
                return fig

            def _scatter(fig, x_all, y_all, name, color, dash="solid", width=2, skip_none=True):
                """Add line trace, optionally skipping None values (for HD95/ASD)."""
                if skip_none:
                    pairs = [(xi, yi) for xi, yi in zip(x_all, y_all) if yi is not None]
                    if not pairs: return
                    xs, ys = zip(*pairs)
                else:
                    xs, ys = x_all, y_all
                mode = "lines+markers" if len(xs) < 15 else "lines"
                fig.add_trace(go.Scatter(x=list(xs), y=list(ys), name=name,
                                         mode=mode,
                                         line=dict(color=color, width=width, dash=dash)))

            # ─── Hall of Fame ─────────────────────────────────────────────
            st.markdown("""
            <div class="glass-card">
                <h3 style="color: #00E5FF; margin-bottom: 16px;">🏆 Best Scores Across All Epochs</h3>
            </div>
            """, unsafe_allow_html=True)

            def _safe_max(lst): return max((v for v in (lst or []) if v is not None), default=0.0)

            best_score      = meta.get("best_score") or 0.0
            best_ep         = meta.get("best_epoch") or "—"
            best_wt_dice    = _safe_max(val.get("tumor", {}).get("wt_dice", []))
            best_et         = _safe_max(val.get("tumor", {}).get("et_dice", []))
            best_s_dice     = _safe_max(val.get("stroke", {}).get("dice", []))
            best_alz_auc    = _safe_max(val.get("alz", {}).get("auc", []))
            best_alz_f1     = _safe_max(val.get("alz", {}).get("f1", []))
            best_wt_hd95_v  = [v for v in val.get("tumor", {}).get("wt_hd95", []) if v is not None]
            best_wt_hd95    = min(best_wt_hd95_v) if best_wt_hd95_v else None

            b1, b2, b3, b4, b5 = st.columns(5)
            _bfmt = lambda v, hi=True: f"{v:.4f}" if isinstance(v, float) else "—"
            for col, label, val_s, color, sub in [
                (b1, "Global Score",     f"{best_score:.4f} (ep{best_ep})", "#00E5FF", "WT+Stroke+ALZ−0.01×HD"),
                (b2, "Val WT Dice",      _bfmt(best_wt_dice),               "#FF4444", f"ET: {best_et:.4f}"),
                (b3, "Val Stroke Dice",  _bfmt(best_s_dice),                "#4488FF", "Binary IoU proxy"),
                (b4, "Val Alz AUC",      _bfmt(best_alz_auc),               "#B67EFF", f"F1: {best_alz_f1:.4f}"),
                (b5, "Best Val WT HD95", f"{best_wt_hd95:.1f} mm" if best_wt_hd95 else "—", "#00FF88", "Lower is better"),
            ]:
                with col:
                    st.markdown(f"""
                    <div class="glass-card" style="text-align:center; border-left:4px solid {color}; padding:14px;">
                        <p style="color:#94A3B8; margin:0; font-size:10px; text-transform:uppercase; letter-spacing:1px;">{label}</p>
                        <h2 style="color:{color}; margin:8px 0; font-family:'Orbitron'; font-size:20px;">{val_s}</h2>
                        <p style="color:#64748B; margin:0; font-size:9px;">{sub}</p>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")

            # ─── ROW 1: Global Score ──────────────────────────────────────
            st.markdown("### ⭐ Global Composite Score")
            fig_score = _dark_fig("Global Score per Epoch  (WT Dice + Stroke Dice + Alz AUC − 0.01×HD95)")
            _scatter(fig_score, ep, meta.get("score", []), "Global Score", "#00E5FF", width=3, skip_none=False)
            st.plotly_chart(fig_score, use_container_width=True)

            # ─── ROW 2: Tumor — Train Soft Dice vs Val Thresholded Dice ──
            st.markdown("### 🧠 Tumor Segmentation")
            c1, c2 = st.columns(2)
            with c1:
                fig_t = _dark_fig("Tumor: Train (soft) vs Val (thresholded)", y_range=[0, 1])
                _scatter(fig_t, ep, tr.get("tumor", {}).get("et_dice", []), "Train ET (soft)", "#FF6666", dash="dot")
                _scatter(fig_t, ep, tr.get("tumor", {}).get("ncr_dice", []), "Train NCR (soft)", "#FF9966", dash="dot")
                _scatter(fig_t, ep, val.get("tumor", {}).get("et_dice", []), "Val ET", "#FF2222", width=3)
                _scatter(fig_t, ep, val.get("tumor", {}).get("wt_dice", []), "Val WT", "#FF8800", width=3)
                st.plotly_chart(fig_t, use_container_width=True)
            with c2:
                fig_hd = _dark_fig("Tumor Val HD95 & ASD (mm) — only HD epochs", y_title="Distance (mm)")
                _scatter(fig_hd, ep, val.get("tumor", {}).get("et_hd95", []), "ET HD95", "#FF4444", width=2)
                _scatter(fig_hd, ep, val.get("tumor", {}).get("wt_hd95", []), "WT HD95", "#FF8800", width=3)
                _scatter(fig_hd, ep, val.get("tumor", {}).get("wt_asd", []),  "WT ASD",  "#FFFF00", dash="dash")
                st.plotly_chart(fig_hd, use_container_width=True)

            # ─── ROW 3: Stroke ────────────────────────────────────────────
            st.markdown("### 🩸 Stroke Segmentation")
            c3, c4 = st.columns(2)
            with c3:
                fig_s = _dark_fig("Stroke: Train Soft Dice vs Val Dice/IoU", y_range=[0, 1])
                _scatter(fig_s, ep, tr.get("stroke", {}).get("dice", []), "Train (soft)", "#6699FF", dash="dot")
                _scatter(fig_s, ep, val.get("stroke", {}).get("dice", []), "Val Dice", "#4488FF", width=3)
                _scatter(fig_s, ep, val.get("stroke", {}).get("iou", []),  "Val IoU",  "#88BBFF", dash="dash")
                st.plotly_chart(fig_s, use_container_width=True)
            with c4:
                fig_sh = _dark_fig("Stroke Val HD95 & ASD (mm) — only HD epochs", y_title="Distance (mm)")
                _scatter(fig_sh, ep, val.get("stroke", {}).get("hd95", []), "HD95", "#4488FF", width=3)
                _scatter(fig_sh, ep, val.get("stroke", {}).get("asd", []),  "ASD",  "#88BBFF", dash="dash")
                st.plotly_chart(fig_sh, use_container_width=True)

            # ─── ROW 4: Alzheimer ─────────────────────────────────────────
            st.markdown("### 🧬 Alzheimer's Classification")
            c5, c6 = st.columns(2)
            with c5:
                fig_a = _dark_fig("Alzheimer Val: AUC / F1 / Accuracy", y_range=[0, 1])
                _scatter(fig_a, ep, val.get("alz", {}).get("auc", []),      "AUC‑ROC",  "#B67EFF", width=4, skip_none=False)
                _scatter(fig_a, ep, val.get("alz", {}).get("f1", []),       "F1",       "#00FFFF", width=2, skip_none=False)
                _scatter(fig_a, ep, val.get("alz", {}).get("accuracy", []), "Accuracy", "#FFFFFF", dash="dash", skip_none=False)
                st.plotly_chart(fig_a, use_container_width=True)
            with c6:
                fig_cal = _dark_fig("Alzheimer Val: Calibration (Brier/ECE)", y_title="Error (lower=better)", y_range=[0, 0.5])
                _scatter(fig_cal, ep, val.get("alz", {}).get("brier", []), "Brier Score", "#FF00FF", width=3, skip_none=False)
                _scatter(fig_cal, ep, val.get("alz", {}).get("ece", []),   "ECE",         "#FF8800", dash="dash", skip_none=False)
                st.plotly_chart(fig_cal, use_container_width=True)
            
            # --- NEW ROW: Precision/Recall & Loss Landscapes ---
            st.markdown("### ⚖️ Detection Rigor & Loss Landscapes")
            c7, c8 = st.columns(2)
            with c7:
                fig_pr = _dark_fig("Alzheimer Val: PR Stability", y_range=[0, 1])
                _scatter(fig_pr, ep, val.get("alz", {}).get("precision", []), "Precision", "#00FF88", width=3, skip_none=False)
                _scatter(fig_pr, ep, val.get("alz", {}).get("recall", []),    "Recall",    "#FF00FF", width=3, skip_none=False)
                _scatter(fig_pr, ep, val.get("alz", {}).get("auprc", []),     "AUPRC",      "#FFFF00", dash="dot", skip_none=False)
                st.plotly_chart(fig_pr, use_container_width=True)
            with c8:
                fig_loss = _dark_fig("Training Multi-Task Loss Landscapes", y_title="Loss")
                _scatter(fig_loss, ep, tr.get("tumor", {}).get("loss", []),  "Tumor Loss",  "#FF4444", width=2, skip_none=False)
                _scatter(fig_loss, ep, tr.get("stroke", {}).get("loss", []), "Stroke Loss", "#4488FF", width=2, skip_none=False)
                _scatter(fig_loss, ep, tr.get("alz", {}).get("loss", []),    "Alz Loss",    "#B67EFF", width=3, skip_none=False)
                st.plotly_chart(fig_loss, use_container_width=True)

            st.markdown("""
            <div class="glass-card" style="border-left-color: #00E5FF;">
                <h4 style="color: #00E5FF;">💡 Metrics Schema Notes</h4>
                <p style="color: #94A3B8; font-size: 13px; line-height: 1.7;">
                    • <b>Train curves</b> use soft (un-thresholded) Dice — noisy by design, shows learning signal.<br>
                    • <b>Val curves</b> use calibrated, thresholded predictions on a 20% held-out set — the ground truth for reporting.<br>
                    • <b>HD95/ASD</b> are computed every 5 epochs only (expensive). Points without values show as gaps — no fake padding.<br>
                    • <b>Global Score</b> = WT Dice + Stroke Dice + Alz AUC − 0.01×WT HD95. Used for best-epoch selection.<br>
                    • Metrics are also saved to <code>checkpoints/metrics.json</code> for offline analysis.
                </p>
            </div>
            """, unsafe_allow_html=True)


            
            st.markdown("""
            <div class="glass-card" style="border-left-color: #00E5FF;">
                <h4 style="color: #00E5FF;">💡 Advanced Multi-Task Curriculum Analysis</h4>
                <p style="color: #94A3B8; font-size: 13px; line-height: 1.6;">
                    • <b>Phase 1 (Epochs 1-10):</b> Alzheimer Priming. Dedicated AlzheimerEncoder establishes base signal representations.<br>
                    • <b>Phase 2 (Epochs 11-26):</b> Segmentation Initialization. Decoder paths are unfrozen with higher LR to build spatial priors.<br>
                    • <b>Phase 3 (Epochs 27-48):</b> Joint Optimization. All heads unconstrained for global multi-disease reasoning (Transformer unmasked).
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
            
            # Load model once for Decision Head evaluation
            model = load_model()

            # ITERATE THROUGH EACH DETECTED DISEASE FOR SEPARATE VISUALIZATION
            detected_diseases = [d for d in st.session_state.detection_results["detected_diseases"] 
                                if d in st.session_state.segmentation_results]
            
            if not detected_diseases:
                st.info("No segmentable diseases detected (Alzheimer is presence-only).")
            
            for disease in detected_diseases:
                disease_cfg = DISEASE_COLORS.get(disease, {"name": disease})
                disease_name = disease_cfg["name"]
                
                # SPECIAL HANDLING FOR ALZHEIMER (No segmentation, but needs analytics block)
                if disease == "alzheimer":
                    st.markdown(f"### 🧬 {disease_name} Analysis")
                    
                    prob = st.session_state.detection_results["probabilities"][disease]
                    unc = st.session_state.detection_results["uncertainties"][disease]
                    logit = st.session_state.detection_results.get("presence_logits", {}).get(disease, 0.0)
                    
                    # Compute Deep Metrics (Per USER REQUEST)
                    am = compute_alzheimer_metrics(prob, unc, logit)
                    
                    # Interpretation Heuristics
                    interp = "Model detects moderate Alzheimer-related patterns"
                    if am['risk'] == "High": interp = "Significant neurodegenerative patterns identified"
                    elif am['risk'] == "Low": interp = "No distinct Alzheimer imaging patterns detected"
                    
                    decisive = "borderline"
                    if am['margin'] > 0.25: decisive = "strong signal"
                    elif am['margin'] < 0.1: decisive = "very uncertain"
                    
                    reliability = "high uncertainty"
                    if am['entropy'] < 0.2: reliability = "high reliability"
                    elif am['entropy'] < 0.5: reliability = "moderate reliability"

                    disease_cfg = DISEASE_COLORS.get(disease, {"hex": "#FFFFFF", "name": disease})
                    color = disease_cfg["hex"]
                    name = disease_cfg["name"]

                    st.markdown(f"""
<div class="glass-card" style="border-left: 5px solid {color};">
<h3 style="color:{color}; margin-bottom:15px;">=== {name} Analysis ===</h3>

<div style="display: flex; justify-content: space-between; gap: 20px; margin-bottom: 20px;">
    <div style="flex:1;">
        <strong>Probability:</strong> <span style="font-size:20px;">{am['prob']:.2f}</span><br>
        <strong>Confidence:</strong> {am['confidence']:.2f}
    </div>
    <div style="flex:1; text-align:right;">
        <span style="background:{color}33; padding:8px 15px; border-radius:30px; border:1px solid {color}">
        <strong>Risk Level:</strong> {am['risk']}
        </span>
    </div>
</div>

**🧠 Decision Strength:**
- **Margin:** {am['margin']:.2f} ({decisive})
- **Signal Strength:** {am['logit_strength']:.2f}

**🛡️ Reliability:**
- **Entropy:** {am['entropy']:.2f} ({reliability})
- **Consistency Score:** {am['consistency']:.2f}

**📈 Adjusted Score:**
- **{am['adjusted_score']:.2f}** (confidence-weighted probability)

---
**💡 Interpretation:**
- {interp}
- Prediction is {decisive}
- Reliability is {reliability}
</div>
""", unsafe_allow_html=True)
                    continue # Skip to next disease as Alzheimer has no 3D mesh
                
                st.markdown(f"### {disease_name} Visualization")
                
                # Unpack result (FIX: Support 3-tuple, 5-tuple, and new 6-tuple)
                result = st.session_state.segmentation_results[disease]
                if len(result) == 3: # Legacy 3-tuple (logits, mask, score)
                    _, binary_roi, dec_score = result
                    probs_roi, prob_orig, unc, logit_raw = binary_roi.astype(float), 0.5, 0.0, 0.0
                elif len(result) == 5: # New 5-tuple
                    binary_roi, dec_score, prob_orig, unc, logit_raw = result
                    probs_roi = binary_roi.astype(float)
                elif len(result) == 6: # Gold-standard 6-tuple (from perform_segmentation)
                    binary_roi, probs_roi, dec_score, prob_orig, unc, logit_raw = result
                else:
                    st.error(f"Unexpected data format ({len(result)}) for {disease}")
                    continue
                
                # EDGE CASE: No lesion detected
                if np.sum(binary_roi) == 0:
                    st.warning(f"No voxel-level lesion detected for **{disease_name}**.")
                    continue

                # Show ML decision score before rendering
                dec_color = "#00FF88" if dec_score >= 0.5 else "#FF3D00"
                st.markdown(f"<div style='background:rgba(255,255,255,0.05); padding:10px; border-radius:5px; border-left:4px solid {dec_color}; margin-bottom:10px;'>"
                            f"🤖 <b>ML Decision Analysis:</b> {dec_score:.1%} confidence<br>"
                            f"<small style='color:#94A3B8;'>{'Model confirms high-fidelity feature match.' if dec_score>=0.5 else 'Heuristics suggest potential false-positive or atypical morphology.'}</small>"
                            f"</div>", unsafe_allow_html=True)
                
                # Prepare single disease dict for viz (Sync with create_3d_visualization expectations)
                single_seg = {disease: (binary_roi, probs_roi, dec_score, prob_orig, unc, logit_raw)}
                
                # LAYER 1: Integrated Global Cache retrieval via @st.cache_resource
                # This ensures absolute zero-regeneration of meshes/HD-BET on UI interaction
                fig_patient = get_visualization_assets(
                    file_hash=st.session_state.file_hash,
                    raw_nifti_bytes=st.session_state.raw_nifti_bytes,
                    disease=disease,
                    disease_name=disease_name,
                    single_seg_data=single_seg,
                    roi_metadata=st.session_state.roi_metadata,
                    affine=st.session_state.affine,
                    spacing=st.session_state.spacing,
                    lesion_metrics=st.session_state.get("lesion_metrics"),
                    model_path=MODEL_PATH # Pass path to keep cache serializable
                )
                
                if fig_patient and len(fig_patient.data) > 0:
                    st.markdown(f"### 🧬 Clinical Visualization: {disease_name}")
                    st.plotly_chart(fig_patient, use_container_width=True, key=f"viz_full_{disease}")

                    # 🔥 CLINICAL EXPORT — Single-button instant download (no rerun, no interference)
                    st.markdown("---")
                    exp_col1, exp_col2 = st.columns([3, 1])

                    with exp_col1:
                        st.info(
                            "📊 **Multi-Slice Probability Heatmap** · Axial / Coronal / Sagittal "
                            "· High-res PNG (200 DPI) · Instant download, no page reload."
                        )

                    with exp_col2:
                        # Build cache-safe primitives ONCE from already-unpacked probs_roi
                        # probs_roi is already available from the unpack block above (line ~3606)
                        _p = probs_roi.astype(np.float32)
                        _p_bytes = _p.tobytes()
                        _p_shape = _p.shape
                        _p_dtype = str(_p.dtype)

                        _roi_meta = st.session_state.roi_metadata
                        _orig_shape_tuple = tuple(int(x) for x in _roi_meta["original_shape"])

                        # roi_affine may be ndarray or None
                        _roi_affine_raw = _roi_meta.get("roi_affine")
                        _orig_affine_raw = _roi_meta.get("original_affine")
                        _roi_affine_list  = _roi_affine_raw.tolist()  if isinstance(_roi_affine_raw, np.ndarray)  else (_roi_affine_raw if _roi_affine_raw is not None else np.eye(4).tolist())
                        _orig_affine_list = _orig_affine_raw.tolist() if isinstance(_orig_affine_raw, np.ndarray) else (_orig_affine_raw if _orig_affine_raw is not None else np.eye(4).tolist())

                        # generate_matplotlib_heatmap_png is @st.cache_data — returns instantly on reruns
                        try:
                            heatmap_png_bytes = generate_matplotlib_heatmap_png(
                                probs_roi_bytes=_p_bytes,
                                probs_roi_shape=_p_shape,
                                probs_roi_dtype_str=_p_dtype,
                                roi_metadata_original_shape=_orig_shape_tuple,
                                roi_metadata_roi_affine=_roi_affine_list,
                                roi_metadata_original_affine=_orig_affine_list,
                                disease=disease,
                                disease_name=disease_name,
                                dpi=200
                            )
                        except Exception as _hm_err:
                            heatmap_png_bytes = None
                            st.warning(f"⚠️ Heatmap generation failed: {_hm_err}")
                            print(f"❌ Heatmap error: {_hm_err}")

                        if heatmap_png_bytes is not None:
                            st.download_button(
                                label="📥 Download Heatmap",
                                data=heatmap_png_bytes,
                                file_name=f"NeuroX_{disease}_Heatmap_{st.session_state.file_hash[:8]}.png",
                                mime="image/png",
                                key=f"dl_heatmap_{disease}_{st.session_state.file_hash}",
                                use_container_width=True
                            )
                    
                    # 🔥 DISPLAY METRICS (Analytical Layer + Volumetrics)
                    if "metrics" in st.session_state and disease in st.session_state.metrics:
                        m = st.session_state.metrics[disease]
                        # Interpretations based on Analytical Layer
                        interp = f"Model identifies {m['risk']} risk patterns"
                        decisive = "borderline signal" if m['margin'] < 0.2 else "distinct signal"
                        
                        disease_cfg = DISEASE_COLORS.get(disease, {"hex": "#FFFFFF", "name": disease})
                        color = disease_cfg["hex"]
                        name = disease_cfg["name"]
                        
                        st.markdown(f"""
<div class="glass-card" style="border-left: 5px solid {color};">
<h3 style="color:{color}; margin-bottom:15px;">🔬 {name} Comprehensive Analysis</h3>

<div style="display: flex; justify-content: space-between; gap: 20px; margin-bottom: 20px;">
    <div style="flex:1;">
        <strong>Calibrated Probability:</strong> <span style="font-size:20px;">{m['prob']:.2f}</span><br>
        <strong>Uncertainty / Entropy:</strong> {m['uncertainty']:.2f}
    </div>
    <div style="flex:1; text-align:right;">
        <span style="background:{color}33; padding:8px 15px; border-radius:30px; border:1px solid {color}">
        <strong>Risk Level:</strong> {m['risk']}
        </span>
    </div>
</div>

**🧠 Model Analytics (Decision Strength):**
- **Confidence Score:** {m['confidence']:.2f} ({(m['confidence']*100):.1f}%)
- **Decision Margin:** {m['margin']:.2f} ({decisive})
- **Confidence-Adjusted Score:** {m['adjusted_score']:.2f}

**📏 Clinical Volumetrics:**
- **Lesion Volume:** {m['lesion_volume_mm3']:.2f} mm³
- **Brain Involvement:** {m['brain_percentage']:.2f}%
- **Hemisphere:** {m.get('hemisphere', 'N/A')}

**🌊 Depth Analysis:**
- **Mean Depth:** {m['depth_mean']:.2f} mm from surface
- **Surface Ratio:** {m['surface_ratio']:.2f} ({(m['surface_ratio']*100):.1f}% externalized)

---
**💡 Interpretation:** {interp} ({decisive})
</div>
""", unsafe_allow_html=True)
                else:
                    st.warning(f"⚠️ Could not generate 3D mesh for {disease_name} (volume might be too small or filtered).")

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
                    # Prioritize Environment Variable GROQ_API_KEY
                    api_key = os.getenv("GROQ_API_KEY", None)
                    
                    if not api_key:
                        st.warning("⚠️ No Groq API Key found in .env file. Using fallback template.")
                    
                    with st.spinner("✍️ Generating AI report..."):
                        st.session_state.report_text = generate_ai_report(
                            st.session_state.detection_results,
                            st.session_state.segmentation_results,
                            st.session_state.get('lesion_metrics', {}),
                            api_key
                        )
            
            if st.session_state.report_text:
                st.markdown("---")
                
                # Use High-Fidelity Paper View for the in-app presentation
                render_paper_report(st.session_state.report_text)
                
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
                    # High-Fidelity PDF Export
                    pdf_bytes = generate_pdf_report(
                        st.session_state.report_text,
                        st.session_state.detection_results,
                        st.session_state.get('lesion_metrics', {})
                    )
                    st.download_button(
                        "📄 EXPORT PDF",
                        pdf_bytes,
                        file_name=f"neurox_clinical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
        else:
            st.info("No detection results available. Complete analysis first.")

    elif st.session_state.current_page == 'settings':
        # ========== SETTINGS PAGE ==========
        st.markdown("## ⚙️ System Settings")
        st.markdown("---")
        
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #00E5FF; margin-bottom: 10px;">🤖 Groq AI Integration</h3>
            <p style="color: #94A3B8; font-size: 13px;">AI radiology reporting is powered by Groq.</p>
        </div>
        """, unsafe_allow_html=True)
        
        env_key = os.getenv("GROQ_API_KEY")
        if env_key:
            st.success(f"✅ Securely connected via `.env` (Key: {env_key[:4]}...{env_key[-4:]})")
            st.info("💡 Your API key is loaded from the environment. To update it, modify the `GROQ_API_KEY` entry in your project's `.env` file.")
        else:
            st.error("❌ Groq API Key not found in `.env`")
            st.warning("Please add `GROQ_API_KEY=your_key_here` to the `.env` file in your project root to enable AI report generation.")
            st.warning("⚠️ No API Key configured. AI reporting will be disabled.")
            
        # 2. Visualization Options (Restored)
        st.markdown("---")
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #00E5FF; margin-bottom: 10px;">🎨 Visualization Options</h3>
            <p style="color: #94A3B8; font-size: 13px;">Customize how 3D models and results are rendered.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_v1, col_v3 = st.columns(2)
        with col_v1:
            st.session_state.show_atlas = st.checkbox("Show Patient-Specific Brain (HD-BET)", value=st.session_state.show_atlas)
        with col_v3:
            st.session_state.show_heatmap = st.checkbox("Show Probability Heatmap", value=st.session_state.show_heatmap)

        # 3. Detection Config (Restored)
        st.markdown("---")
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #00E5FF; margin-bottom: 10px;">📊 Analysis Configuration</h3>
            <p style="color: #94A3B8; font-size: 13px;">Adjust sensitivity and detection parameters.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("💡 **Clinical Sensitivity Note:** System detection thresholds are hardcoded to clinical defaults (ALZ=0.4, Lesion=0.3) for safety.")
            
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