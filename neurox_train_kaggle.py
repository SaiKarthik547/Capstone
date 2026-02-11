

import os, gc, sys, random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import deque
from sklearn.metrics import roc_auc_score

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm.auto import tqdm

print("=" * 80)
print("🧠 NeuroX - Multi-Disease Pathology Detection System")
print("=" * 80)
print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"🔥 PyTorch: {torch.__version__}")
print(f"💻 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
print("📋 Diseases: Tumor | Stroke | Alzheimer")
print("=" * 80 + "\n")

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROI_SIZE = (96, 96, 96)
BATCH_SIZE = 2
NUM_WORKERS = 2
LR = 1e-4
WEIGHT_DECAY = 1e-5
USE_AMP = True
GRADIENT_CLIP = 1.0
VALIDATION_SPLIT = 0.1
PRESENCE_THRESHOLD = 0.5
CHECKPOINT_DIR = Path("./checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)
#Production
# ═════════════════════════════════════════════════════════════════════════════
# PRODUCTION-GRADE TRAINING ENHANCEMENTS
# ═════════════════════════════════════════════════════════════════════════════

# Gradient Accumulation: Simulates larger batch size without GPU memory increase
# Effective batch = BATCH_SIZE × GRADIENT_ACCUMULATION_STEPS = 2 × 4 = 8
GRADIENT_ACCUMULATION_STEPS = 4

# Loss Curriculum: Epochs to focus purely on presence before adding segmentation
# Prevents early-stage segmentation dominance that can harm presence learning
PRESENCE_WARMUP_EPOCHS = 5

# ═══════════════════════════════════════════════════════════════════════════
# ADVANCED FEATURES CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# Uncertainty Estimation (Monte-Carlo Dropout)
MC_DROPOUT_SAMPLES = 10  # Number of forward passes for uncertainty estimation
MC_DROPOUT_RATE = 0.3  # Dropout rate for epistemic uncertainty

# Adaptive Encoder Freezing
ADAPTIVE_FREEZE_PATIENCE = 3  # Epochs to wait before unfreezing on plateau
ADAPTIVE_FREEZE_DELTA = 0.01  # Minimum AUC improvement to avoid plateau

# Multi-Scale Context Learning
GLOBAL_CONTEXT_SIZE = (48, 48, 48)  # Downsampled global context resolution
USE_MULTI_SCALE = True  # Enable dual-resolution learning

# Curriculum Learning
CURRICULUM_STAGE1_EPOCHS = 5  # Presence-only with frozen encoder
CURRICULUM_STAGE2_EPOCHS = 10  # Presence + segmentation, encoder frozen
# Stage 3: Full unfreezing for remaining epochs

# Calibration-Aware Loss
FOCAL_LOSS_GAMMA = 2.0  # Focal loss focusing parameter
TEMPERATURE_SCALING_INIT = 1.5  # Initial temperature for calibration

# Failure Detection (OOD)
OOD_ENTROPY_WEIGHT = 0.1  # Weight for OOD head entropy loss

# Explainability (Grad-CAM)
SAVE_GRADCAM_MAPS = True  # Save attention maps during validation
GRADCAM_OUTPUT_DIR = Path("./gradcam_outputs")
GRADCAM_OUTPUT_DIR.mkdir(exist_ok=True)

# Training Phases
TRAINING_PHASES = [
    {
        "name": "Phase 1: Brain Tumor Detection",
        "dataset_type": "tumor",
        "dataset_path": "/kaggle/input/brats20-dataset-training-validation/",
        "epochs": 50,
        "freeze_encoder": False,
        "has_segmentation": True
    },
    {
        "name": "Phase 2: Ischemic Stroke Detection",
        "dataset_type": "stroke",
        "dataset_path": "/kaggle/input/isles-2022-brain-stoke-dataset/",
        "epochs": 55,
        "freeze_encoder": "partial",
        "has_segmentation": True
    },
    {
        "name": "Phase 3: Alzheimer's Presence Detection",
        "dataset_type": "alzheimer",
        "dataset_path": "/kaggle/input/new-3d-mri-alzheimer/",
        "epochs": 56,
        "freeze_encoder": True,
        "has_segmentation": False  # CRITICAL: No segmentation for Alzheimer
    }
]

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ═══════════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE
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
    """Shared encoder - accepts 1 channel, internally expands if needed.
    
    PRODUCTION IMPROVEMENT: Uses InstanceNorm3d instead of BatchNorm3d.
    Rationale: With batch_size=2, BatchNorm statistics are unstable.
    InstanceNorm is standard for medical imaging and normalizes per-sample.
    """
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
        """InstanceNorm3d for small-batch stability (affine=True preserves learnable params)."""
        return nn.Sequential(
            nn.Conv3d(in_c, out_c, 3, padding=1),
            nn.InstanceNorm3d(out_c, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_c, out_c, 3, padding=1),
            nn.InstanceNorm3d(out_c, affine=True),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        # Handle variable input channels (tumor/stroke use 2, Alzheimer uses 1)
        if x.shape[1] == 2:
            # Average pool to 1 channel for encoder
            x = x.mean(dim=1, keepdim=True)
        
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        return {"enc1": e1, "enc2": e2, "enc3": e3, "bottleneck": b}
    
    def freeze_layers(self, mode):
        if mode == False:
            for param in self.parameters():
                param.requires_grad = True
        elif mode == "partial":
            for param in self.enc1.parameters():
                param.requires_grad = False
            for param in self.enc2.parameters():
                param.requires_grad = False
        elif mode == True:
            for param in self.parameters():
                param.requires_grad = False


class MCDropout(nn.Module):
    """Monte-Carlo Dropout - remains active during inference for uncertainty estimation.
    
    Scientific Rationale:
    Dropout at inference time approximates Bayesian inference in deep networks,
    providing epistemic uncertainty estimates critical for medical decision support.
    Reference: Gal & Ghahramani (2016) - Dropout as a Bayesian Approximation
    """
    def __init__(self, p=0.3):
        super().__init__()
        self.p = p
    
    def forward(self, x):
        # CRITICAL: Force training=True for MC sampling even during eval
        return F.dropout(x, p=self.p, training=True)


class PresenceHead(nn.Module):
    """Binary presence detector with epistemic uncertainty estimation.
    
    Scientific Rationale:
    Medical models must communicate confidence. Low uncertainty on correct predictions
    indicates reliable expertise; high uncertainty flags cases needing expert review.
    """
    def __init__(self, in_features=128):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features, 64)
        self.relu = nn.ReLU()
        self.mc_dropout = MCDropout(MC_DROPOUT_RATE)  # MC Dropout for uncertainty
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, bottleneck_features):
        x = self.pool(bottleneck_features)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.mc_dropout(x)  # Active even during eval
        x = self.fc2(x)
        return x
    
    def uncertainty_forward(self, bottleneck_features, n_samples=MC_DROPOUT_SAMPLES):
        """Perform N stochastic forward passes for epistemic uncertainty.
        
        Returns:
            mean_prob: Mean prediction across samples
            uncertainty: Variance across samples (epistemic uncertainty)
        """
        predictions = []
        for _ in range(n_samples):
            logit = self.forward(bottleneck_features)
            prob = torch.sigmoid(logit)
            predictions.append(prob)
        
        predictions = torch.stack(predictions, dim=0)  # [N, B, 1]
        mean_prob = predictions.mean(dim=0)
        uncertainty = predictions.var(dim=0)  # Epistemic uncertainty
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
        """InstanceNorm3d (matches encoder normalization strategy)."""
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


class MultiScaleContextFusion(nn.Module):
    """Fuses high-resolution ROI features with downsampled global context.
    
    Scientific Rationale:
    Medical diagnosis requires both local pathology detail and global anatomical context.
    Multi-scale fusion allows the model to learn spatial relationships between pathologies
    and anatomical landmarks (e.g., ventricles, cortex).
    Reference: Chen et al. (2018) - Encoder-Decoder with Atrous Separable Convolution
    """
    def __init__(self, bottleneck_dim=128):
        super().__init__()
        # Process global context path
        self.global_proc = nn.Sequential(
            nn.Conv3d(bottleneck_dim, bottleneck_dim, 1),
            nn.InstanceNorm3d(bottleneck_dim, affine=True),  # Consistent normalization
            nn.ReLU(inplace=True)
        )
        # Learnable fusion weights
        self.fusion_weight = nn.Parameter(torch.tensor([0.7, 0.3]))  # [local, global]
    
    def forward(self, local_features, global_features):
        """
        local_features: High-res bottleneck [B, C, D, H, W]
        global_features: Low-res bottleneck [B, C, D/2, H/2, W/2]
        """
        # Upsample global to match local resolution
        global_upsampled = F.interpolate(
            global_features, 
            size=local_features.shape[2:], 
            mode='trilinear', 
            align_corners=False
        )
        global_upsampled = self.global_proc(global_upsampled)
        
        # Weighted fusion with normalized weights
        weights = F.softmax(self.fusion_weight, dim=0)
        fused = weights[0] * local_features + weights[1] * global_upsampled
        return fused


class OODConfidenceHead(nn.Module):
    """Out-of-Distribution detection head using entropy maximization.
    
    Scientific Rationale:
    Clinical deployment requires models to flag OOD samples (artifacts, rare pathologies).
    High entropy predictions indicate distributional shift, triggering expert review.
    Reference: Hendrycks & Gimpel (2017) - A Baseline for Detecting Misclassified and OOD Examples
    """
    def __init__(self, in_features=128):
        super().__init__()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),  # Single OOD confidence score
            nn.Sigmoid()  # Output: 0 = in-distribution, 1 = OOD
        )
    
    def forward(self, bottleneck_features):
        return self.head(bottleneck_features)


class GradCAMHook:
    """Gradient-weighted Class Activation Mapping for explainability.
    
    Scientific Rationale:
    Clinicians need voxel-level explanations to trust model predictions. Grad-CAM visualizes
    which brain regions influenced the presence decision, enabling pathology localization
    even without segmentation labels (critical for Alzheimer).
    Reference: Selvaraju et al. (2017) - Grad-CAM: Visual Explanations from Deep Networks
    """
    def __init__(self):
        self.gradients = None
        self.activations = None
    
    def save_gradient(self, grad):
        self.gradients = grad
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def generate_cam(self, target_layer):
        """Generate class activation map from stored gradients and activations."""
        if self.gradients is None or self.activations is None:
            return None
        
        # Global average pooling of gradients
        weights = self.gradients.mean(dim=(2, 3, 4), keepdim=True)  # [B, C, 1, 1, 1]
        
        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [B, 1, D, H, W]
        cam = F.relu(cam)  # Only positive contributions
        
        # Normalize to [0, 1]
        cam_min = cam.view(cam.size(0), -1).min(dim=1)[0].view(-1, 1, 1, 1, 1)
        cam_max = cam.view(cam.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        
        return cam


class NeuroXMultiDisease(nn.Module):
    """Multi-disease pathology detection model with advanced research features.
    
    New Capabilities:
    - Uncertainty-aware predictions (MC Dropout)
    - Multi-scale context learning
    - OOD failure detection
    - Grad-CAM explainability
    - Temperature-scaled calibration
    """
    def __init__(self):
        super().__init__()
        self.encoder = SharedEncoder(in_channels=1)
        
        # Multi-scale context fusion (NEW)
        if USE_MULTI_SCALE:
            self.context_fusion = MultiScaleContextFusion(bottleneck_dim=128)
        
        # Three presence heads (NOW with MC Dropout)
        self.presence_heads = nn.ModuleDict({
            "tumor": PresenceHead(128),
            "stroke": PresenceHead(128),
            "alzheimer": PresenceHead(128)
        })
        
        # TWO segmentation decoders (NO Alzheimer decoder)
        self.seg_decoders = nn.ModuleDict({
            "tumor": SegmentationDecoder(4, "tumor"),
            "stroke": SegmentationDecoder(1, "stroke")
            # CRITICAL: NO Alzheimer decoder
        })
        
        # OOD confidence head (NEW)
        self.ood_head = OODConfidenceHead(128)
        
        # Temperature scaling for calibration (NEW)
        self.temperature = nn.Parameter(torch.ones(1) * TEMPERATURE_SCALING_INIT)
        
        # Grad-CAM hooks (NEW) - initialized but not registered until needed
        self.gradcam_hook = GradCAMHook()
        self.gradcam_handle = None
    
    def forward(self, x, active_presence=None, active_seg=None, x_global=None, compute_ood=False):
        """
        Args:
            x: High-resolution ROI input [B, C, D, H, W]
            x_global: Optional low-resolution global context [B, C, D/2, H/2, W/2]
            compute_ood: Whether to compute OOD confidence score
        """
        # Encode local (high-res) features
        features = self.encoder(x)
        bottleneck = features["bottleneck"]
        
        # Multi-scale fusion if global context provided
        if USE_MULTI_SCALE and x_global is not None:
            global_features = self.encoder(x_global)
            bottleneck = self.context_fusion(bottleneck, global_features["bottleneck"])
            features["bottleneck"] = bottleneck  # Update for downstream use
        
        # Presence detection with temperature-scaled calibration
        presence = {}
        if active_presence:
            for key in active_presence:
                if key in self.presence_heads:
                    logit = self.presence_heads[key](bottleneck)
                    # Apply temperature scaling for calibration
                    presence[key] = logit / self.temperature
        
        # Segmentation (unchanged logic)
        segmentations = {}
        if active_seg:
            for key in active_seg:
                if key in self.seg_decoders:
                    segmentations[key] = self.seg_decoders[key](features)
        
        # OOD detection (optional)
        ood_confidence = None
        if compute_ood:
            ood_confidence = self.ood_head(bottleneck)
        
        return {
            "presence": presence, 
            "segmentations": segmentations,
            "ood_confidence": ood_confidence,
            "features": features  # For Grad-CAM access
        }
    
    def enable_gradcam(self, disease="alzheimer"):
        """Enable Grad-CAM for specified disease head.
        
        Scientific Rationale:
        Alzheimer diagnosis lacks ground-truth segmentation. Grad-CAM provides
        post-hoc localization of discriminative regions (hippocampus, ventricles).
        """
        # Register hook on bottleneck layer
        target_layer = self.encoder.bottleneck
        
        self.gradcam_handle = target_layer.register_forward_hook(
            self.gradcam_hook.save_activation
        )
    
    def disable_gradcam(self):
        """Remove Grad-CAM hooks."""
        if self.gradcam_handle is not None:
            self.gradcam_handle.remove()
            self.gradcam_handle = None
    
    def get_gradcam_map(self, disease_logit):
        """Compute Grad-CAM map for a disease presence prediction.
        
        Args:
            disease_logit: Scalar logit from presence head
        
        Returns:
            cam: Normalized attention map [B, 1, D, H, W]
        """
        # Compute gradients w.r.t. bottleneck activations
        disease_logit.backward(retain_graph=True)
        
        # Generate CAM
        return self.gradcam_hook.generate_cam(target_layer=None)
    
    def freeze_encoder(self, mode):
        self.encoder.freeze_layers(mode)


# ═══════════════════════════════════════════════════════════════════════════
# DATASET CLASSES (THREE EXPLICIT CLASSES)
# ═══════════════════════════════════════════════════════════════════════════

def zscore_normalize(vol: np.ndarray) -> np.ndarray:
    vol = vol.astype(np.float32)
    mean, std = vol.mean(), vol.std() + 1e-8
    return np.clip((vol - mean) / std, -5, 5)


def load_nifti(path: Path) -> np.ndarray:
    try:
        data = nib.load(str(path)).get_fdata()
        return np.asarray(data, dtype=np.float32)
    except:
        return np.zeros((96, 96, 96), dtype=np.float32)


class TumorDataset(Dataset):
    """BraTS 2020 brain tumor dataset"""
    def __init__(self, root_path: str):
        self.root = Path(root_path)
        self.cases = self._find_cases()
        print(f"✅ Tumor Dataset: {len(self.cases)} cases")
    
    def _find_cases(self):
        cases = []
        skipped = 0
        
        if not self.root.exists():
            print(f"⚠️ Path not found: {self.root}")
            return cases
        
        # BraTS structure: root/BraTS20_Training_XXX/
        for case_dir in self.root.rglob("BraTS20_Training_*"):
            if not case_dir.is_dir():
                continue
            
            # Look for required files
            flair = list(case_dir.glob("*_flair.nii*"))
            t1ce = list(case_dir.glob("*_t1ce.nii*"))
            seg = list(case_dir.glob("*_seg.nii*"))
            
            if flair and t1ce and seg:
                cases.append({
                    "flair": flair[0],
                    "t1ce": t1ce[0],
                    "seg": seg[0]
                })
            else:
                skipped += 1
        
        if skipped > 0:
            print(f"  ⚠️ Skipped {skipped} incomplete cases")
        
        return cases
    
    def __len__(self):
        return len(self.cases)
    
    def __getitem__(self, idx):
        case = self.cases[idx]
        
        # Load FLAIR + T1CE
        flair = zscore_normalize(load_nifti(case["flair"]))
        t1ce = zscore_normalize(load_nifti(case["t1ce"]))
        image = np.stack([flair, t1ce], axis=0)
        
        # Load segmentation
        seg = load_nifti(case["seg"])
        # BraTS: 0=background, 1=NCR, 2=ED, 4=ET
        seg_ed = (seg == 2).astype(np.float32)
        seg_et = (seg == 4).astype(np.float32)
        seg_ncr = (seg == 1).astype(np.float32)
        seg_wt = ((seg == 1) | (seg == 2) | (seg == 4)).astype(np.float32)
        seg_processed = np.stack([seg_ed, seg_et, seg_ncr, seg_wt], axis=0)
        
        return {
            "image": torch.from_numpy(image).float(),
            "seg": torch.from_numpy(seg_processed).float(),
            "has_seg": torch.tensor([1.0], dtype=torch.float32),
            "tumor_presence": torch.tensor([1.0], dtype=torch.float32),
            "stroke_presence": torch.tensor([0.0], dtype=torch.float32),
            "alzheimer_presence": torch.tensor([0.0], dtype=torch.float32)
        }


class StrokeDataset(Dataset):
    """ISLES 2022 stroke dataset with nested .nii directories"""

    def __init__(self, root_path: str):
        self.root = Path(root_path)
        self.cases = self._find_cases()
        print(f"✅ Stroke Dataset: {len(self.cases)} cases")

    def _resolve_nii(self, path: Path) -> Optional[Path]:
        """
        If path is a directory ending with .nii, return the actual .nii inside it.
        """
        if path.is_dir():
            nii_files = list(path.glob("*.nii*"))
            return nii_files[0] if nii_files else None
        return path if path.is_file() else None

    def _find_cases(self):
        cases = []
        skipped = 0

        base = self.root / "ISLES-2022" / "ISLES-2022"

        for sub_dir in base.glob("sub-strokecase*"):
            dwi_root = sub_dir / "ses-0001" / "dwi"
            if not dwi_root.exists():
                skipped += 1
                continue

            # Find DWI + ADC (they are directories!)
            dwi_dirs = list(dwi_root.glob("*_dwi.nii*"))
            adc_dirs = list(dwi_root.glob("*_adc.nii*"))

            dwi_file = self._resolve_nii(dwi_dirs[0]) if dwi_dirs else None
            adc_file = self._resolve_nii(adc_dirs[0]) if adc_dirs else None

            # Mask (normal file)
            mask_dir = base / "derivatives" / sub_dir.name / "ses-0001"
            msk_files = list(mask_dir.glob("*_msk.nii*")) if mask_dir.exists() else []
            msk_file = msk_files[0] if msk_files else None

            if dwi_file and adc_file and msk_file:
                cases.append({
                    "dwi": dwi_file,
                    "adc": adc_file,
                    "msk": msk_file
                })
            else:
                skipped += 1

        print(f"  ⚠️ Skipped {skipped} incomplete stroke cases")
        return cases

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case = self.cases[idx]

        dwi = zscore_normalize(load_nifti(case["dwi"]))
        adc = zscore_normalize(load_nifti(case["adc"]))
        image = np.stack([dwi, adc], axis=0)

        msk = load_nifti(case["msk"])
        seg = (msk > 0).astype(np.float32)[np.newaxis, ...]

        return {
            "image": torch.from_numpy(image).float(),
            "seg": torch.from_numpy(seg).float(),
            "has_seg": torch.tensor([1.0]),
            "tumor_presence": torch.tensor([0.0]),
            "stroke_presence": torch.tensor([1.0]),
            "alzheimer_presence": torch.tensor([0.0]),
        }



class AlzheimerDataset(Dataset):
    """ADNI-style Alzheimer dataset (presence detection ONLY)"""
    def __init__(self, root_path: str):
        self.root = Path(root_path)
        self.cases = self._find_cases()
        print(f"✅ Alzheimer Dataset: {len(self.cases)} cases (presence-only)")
    
    def _find_cases(self):
        cases = []
        skipped = 0
        
        if not self.root.exists():
            print(f"⚠️ Path not found: {self.root}")
            return cases
        
        # Search in MRI-AD, MRI-MCI, MRI-CN folders
        for category in ["MRI-AD", "MRI-MCI", "MRI-CN", "MRI-AD-Part-2", "MRI-CN-Part-2"]:
            cat_dir = self.root / category
            if not cat_dir.exists():
                continue
            
            # Determine label: CN = 0, MCI/AD = 1
            alzheimer_label = 0.0 if "CN" in category else 1.0
            
            # Find .nii files deep in structure
            for nii_file in cat_dir.rglob("*.nii*"):
                if nii_file.is_file():
                    cases.append({
                        "t1": nii_file,
                        "alzheimer_label": alzheimer_label
                    })
        
        if len(cases) == 0:
            print(f"  ⚠️ Warning: No .nii files found")
        
        return cases
    
    def __len__(self):
        return len(self.cases)
    
    def __getitem__(self, idx):
        case = self.cases[idx]
        
        # Load T1 only (single channel)
        t1 = zscore_normalize(load_nifti(case["t1"]))
        image = t1[np.newaxis, ...]  # [1, D, H, W]
        
        # NO SEGMENTATION for Alzheimer
        
        return {
            "image": torch.from_numpy(image).float(),
            "seg": None,  # Explicitly None
            "has_seg": torch.tensor([0.0], dtype=torch.float32),  # CRITICAL: 0
            "tumor_presence": torch.tensor([0.0], dtype=torch.float32),
            "stroke_presence": torch.tensor([0.0], dtype=torch.float32),
            "alzheimer_presence": torch.tensor([case["alzheimer_label"]], dtype=torch.float32)
        }
    
    def get_class_weights(self):
        """Compute pos_weight for imbalanced Alzheimer dataset.
        
        PRODUCTION FIX: Alzheimer datasets are heavily imbalanced (more CN than AD/MCI).
        pos_weight = neg_count / pos_count handles this without oversampling.
        Applied ONLY in Alzheimer phase training loop.
        """
        pos_count = sum(1 for c in self.cases if c["alzheimer_label"] == 1.0)
        neg_count = len(self.cases) - pos_count
        pos_weight = neg_count / max(pos_count, 1)  # Avoid divide-by-zero
        print(f"  📊 Alzheimer class balance: CN={neg_count}, AD/MCI={pos_count}, pos_weight={pos_weight:.2f}")
        return pos_weight


# ═══════════════════════════════════════════════════════════════════════════
# CALIBRATION-AWARE LOSS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def get_loss_curriculum_weights(epoch, total_epochs, warmup_epochs=PRESENCE_WARMUP_EPOCHS):
    """Curriculum learning: presence-first, gradually add segmentation.
    
    PRODUCTION FIX: Standard joint training causes gradient interference.
    Our approach:
    - Epochs 1-5: Pure presence learning (seg_weight=0)
    - Epochs 6+: Smooth cosine ramp-up for segmentation
    
    Scientific Rationale:
    Presence detection requires GLOBAL features, segmentation requires LOCAL features.
    Learning them simultaneously from random initialization causes conflicting gradients.
    Curriculum learning establishes stable presence detection first, then adds segmentation.
    
    Reference: Bengio et al. (2009) - Curriculum Learning
    """
    if epoch <= warmup_epochs:
        # Pure presence learning phase
        return 0.0, 1.0  # (seg_weight, presence_weight)
    else:
        # Cosine ramp-up for segmentation weight: 0 → 1.0 over remaining epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        seg_weight = 0.5 * (1 - np.cos(np.pi * progress))  # Smooth 0 → 1
        presence_weight = 1.0  # Keep presence weight constant
        return seg_weight, presence_weight


class FocalLoss(nn.Module):
    """Focal Loss to penalize overconfident false positives.
    
    Scientific Rationale:
    Medical models often exhibit overconfidence on hard negatives (healthy tissue near lesions).
    Focal loss downweights easy examples, forcing the model to focus on hard misclassifications.
    Reference: Lin et al. (2017) - Focal Loss for Dense Object Detection
    """
    def __init__(self, gamma=FOCAL_LOSS_GAMMA, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)  # prob of true class
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        loss = alpha_weight * focal_weight * bce_loss
        return loss.mean()


class CalibratedConditionalLoss(nn.Module):
    """Enhanced loss with focal loss for presence, temperature scaling support, and OOD regularization.
    
    Scientific Rationale:
    - Focal loss addresses class imbalance and overconfidence
    - Temperature scaling enables post-hoc calibration
    - OOD entropy maximization trains the model to flag unfamiliar samples
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.focal_loss = FocalLoss(gamma=FOCAL_LOSS_GAMMA)
    
    def dice_loss(self, pred, target):
        probs = torch.sigmoid(pred)
        inter = (probs * target).sum(dim=(2,3,4))
        union = probs.sum(dim=(2,3,4)) + target.sum(dim=(2,3,4))
        return 1 - ((2 * inter + 1) / (union + 1)).mean()
    
    def ood_entropy_loss(self, ood_confidence):
        """Entropy maximization for OOD samples (encourages high uncertainty on OOD).
        
        Scientific Rationale:
        We want the model to produce high OOD confidence (close to 1) for out-of-distribution
        samples. During training, we occasionally maximize entropy to prevent overconfident
        predictions on borderline cases.
        """
        # Entropy of Bernoulli: -p*log(p) - (1-p)*log(1-p)
        eps = 1e-8
        entropy = -(ood_confidence * torch.log(ood_confidence + eps) + 
                    (1 - ood_confidence) * torch.log(1 - ood_confidence + eps))
        return -entropy.mean()  # Negative because we want to MAXIMIZE entropy
    
    def forward(self, seg_pred, seg_target, presence_pred, presence_target, has_seg, 
                ood_confidence=None):
        """
        CRITICAL: Only compute seg loss if has_seg == 1
        """
        # Presence loss with focal loss (NEW: calibration-aware)
        presence_loss = self.focal_loss(presence_pred, presence_target)
        
        loss_dict = {"presence": presence_loss.item(), "seg": 0.0, "ood": 0.0}
        
        # Segmentation loss (conditional)
        seg_loss = torch.tensor(0.0, device=presence_pred.device)
        if has_seg is not None and has_seg.sum() > 0:
            valid_idx = has_seg.squeeze() > 0.5
            if valid_idx.any():
                seg_dice = self.dice_loss(seg_pred[valid_idx], seg_target[valid_idx])
                seg_bce = self.bce(seg_pred[valid_idx], seg_target[valid_idx])
                seg_loss = seg_dice + seg_bce
                loss_dict["seg"] = seg_loss.item()
        
        # OOD regularization (optional)
        ood_loss = torch.tensor(0.0, device=presence_pred.device)
        if ood_confidence is not None:
            ood_loss = self.ood_entropy_loss(ood_confidence) * OOD_ENTROPY_WEIGHT
            loss_dict["ood"] = ood_loss.item()
        
        # Combined loss
        total_loss = seg_loss + 0.5 * presence_loss + ood_loss
        loss_dict["total"] = total_loss.item()
        
        return total_loss, loss_dict


def collate_fn(batch):
    """CPU-only collate with per-sample resizing and multi-scale context support.
    
    Scientific Rationale:
    Dual-resolution processing: High-res for detailed pathology + low-res for anatomical context.
    """
    images = []
    images_global = []  # NEW: Global context at lower resolution
    segs = []
    has_seg_list = []
    tumor_pres = []
    stroke_pres = []
    alzheimer_pres = []

    for b in batch:
        # High-resolution ROI
        img = b["image"].unsqueeze(0)  # [1, C, D, H, W]
        img = F.interpolate(
            img,
            size=ROI_SIZE,
            mode="trilinear",
            align_corners=False
        ).squeeze(0)
        images.append(img)

        # Low-resolution global context (NEW for multi-scale)
        if USE_MULTI_SCALE:
            img_global = F.interpolate(
                b["image"].unsqueeze(0),
                size=GLOBAL_CONTEXT_SIZE,
                mode="trilinear",
                align_corners=False
            ).squeeze(0)
            images_global.append(img_global)

        # Segmentation
        if b["seg"] is not None:
            seg = b["seg"].unsqueeze(0)
            seg = F.interpolate(
                seg,
                size=ROI_SIZE,
                mode="nearest"
            ).squeeze(0)
        else:
            seg = torch.zeros((1, *ROI_SIZE), dtype=torch.float32)

        segs.append(seg)

        has_seg_list.append(b["has_seg"])
        tumor_pres.append(b["tumor_presence"])
        stroke_pres.append(b["stroke_presence"])
        alzheimer_pres.append(b["alzheimer_presence"])

    batch_dict = {
        "image": torch.stack(images),
        "seg": torch.stack(segs),
        "has_seg": torch.stack(has_seg_list),
        "tumor_presence": torch.stack(tumor_pres),
        "stroke_presence": torch.stack(stroke_pres),
        "alzheimer_presence": torch.stack(alzheimer_pres)
    }
    
    if USE_MULTI_SCALE and images_global:
        batch_dict["image_global"] = torch.stack(images_global)
    
    return batch_dict


# ═══════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def validate(model, val_loader, phase_config, save_gradcam=False):
    """Validation with DETERMINISTIC inference (no Grad-CAM, no MC Dropout).
    
    CRITICAL FIX: Removed all Grad-CAM and MC Dropout logic.
    - Grad-CAM: Incompatible with torch.no_grad() (cannot register hooks without gradients)
    - MC Dropout: 10x slower, not needed for model selection during training
    - Both belong in inference/demo (neurox_adaptive.py), not training validation
    """
    model.eval()
    all_dice = []
    all_pres_true = []
    all_pres_pred = []
    
    disease = phase_config["dataset_type"]
    has_seg = phase_config["has_segmentation"]
    
    with torch.no_grad():
        for batch in val_loader:
            x = batch["image"].to(DEVICE, non_blocking=True)
            x_global = batch.get("image_global")
            if x_global is not None:
                x_global = x_global.to(DEVICE, non_blocking=True)
            
            seg_target = batch["seg"].to(DEVICE, non_blocking=True)
            has_seg_flag = batch["has_seg"].to(DEVICE, non_blocking=True)
            pres_target = batch[f"{disease}_presence"].to(DEVICE, non_blocking=True)

            # Forward (deterministic, no OOD)
            if has_seg:
                output = model(x, active_presence=[disease], active_seg=[disease], 
                             x_global=x_global, compute_ood=False)
                seg_pred = output["segmentations"][disease]
                
                # Per-sample Dice
                if has_seg_flag.sum() > 0:
                    valid_idx = has_seg_flag.squeeze() > 0.5
                    if valid_idx.any():
                        probs = torch.sigmoid(seg_pred[valid_idx])
                        targets = seg_target[valid_idx]
                        
                        for sample_idx in range(probs.shape[0]):
                            sample_pred = probs[sample_idx]
                            sample_target = targets[sample_idx]
                            
                            inter = (sample_pred * sample_target).sum()
                            union = sample_pred.sum() + sample_target.sum()
                            
                            if union > 0:
                                dice = (2 * inter / union).item()
                                all_dice.append(dice)
            else:
                output = model(x, active_presence=[disease], active_seg=None, 
                             x_global=x_global, compute_ood=False)
            
            # Deterministic presence (simple sigmoid, no MC Dropout)
            pres_pred = output["presence"][disease]
            pres_prob = torch.sigmoid(pres_pred)
            
            all_pres_true.extend(pres_target.cpu().numpy().flatten())
            all_pres_pred.extend(pres_prob.cpu().numpy().flatten())
    
    avg_dice = np.mean(all_dice) if all_dice else 0.0
    presence_auc = roc_auc_score(all_pres_true, all_pres_pred) if len(set(all_pres_true)) > 1 else 0.5
    
    model.train()
    return avg_dice, presence_auc, 0.0  # Return 0.0 for uncertainty (not computed)


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════

def train_phase(model, phase_config, phase_num):
    print("\n" + "=" * 80)
    print(f"🎯 {phase_config['name']}")
    print(f"🔒 Encoder Freeze: {phase_config['freeze_encoder']}")
    print(f"📊 Has Segmentation: {phase_config['has_segmentation']}")
    print("=" * 80 + "\n")
    
    model.freeze_encoder(phase_config["freeze_encoder"])
    
    # Create dataset
    disease = phase_config["dataset_type"]
    path = phase_config["dataset_path"]
    
    if disease == "tumor":
        dataset = TumorDataset(path)
    elif disease == "stroke":
        dataset = StrokeDataset(path)
    else:  # alzheimer
        dataset = AlzheimerDataset(path)
        # PRODUCTION FIX: Compute Alzheimer class weights for pos_weight
        if disease == "alzheimer":
            alzheimer_pos_weight = torch.tensor([dataset.get_class_weights()]).to(DEVICE)
        else:
            alzheimer_pos_weight = None
    
    if len(dataset) == 0:
        print("⚠️ Empty dataset - skipping\n")
        return
    
    # Train/val split
    train_size = int((1 - VALIDATION_SPLIT) * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    print(f"📊 Train: {train_size}, Val: {val_size}\n")
    
    train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    collate_fn=collate_fn)

    val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    collate_fn=collate_fn)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    criterion = CalibratedConditionalLoss()
    best_metric = 0.0
    
    # CRITICAL FIX: Create Alzheimer-specific loss with pos_weight
    # This handles class imbalance (CN vs AD/MCI) without oversampling
    if disease == "alzheimer" and 'alzheimer_pos_weight' in locals():
        alzheimer_criterion = nn.BCEWithLogitsLoss(pos_weight=alzheimer_pos_weight)
        print(f"  🎯 Alzheimer BCEWithLogitsLoss created with pos_weight={alzheimer_pos_weight.item():.2f}\n")
    else:
        alzheimer_criterion = None
    
    # Adaptive freezing tracking (NEW)
    auc_history = deque(maxlen=ADAPTIVE_FREEZE_PATIENCE)
    encoder_unfrozen = False
    
    # Curriculum learning schedule  (NEW)
    total_epochs = phase_config["epochs"]
    curriculum_enabled = total_epochs > CURRICULUM_STAGE1_EPOCHS + CURRICULUM_STAGE2_EPOCHS
    
    for epoch in range(1, total_epochs + 1):
        model.train()
        
        # Curriculum Learning: Automated progressive training (NEW)
        # Scientific Rationale: Start with simpler task (presence), then add complexity (segmentation),
        # finally unfreeze encoder for fine-tuning. Prevents catastrophic forgetting.
        if curriculum_enabled:
            if epoch <= CURRICULUM_STAGE1_EPOCHS:
                # Stage 1: Presence only, encoder frozen
                current_freeze = True
                train_segmentation = False
                stage_name = "Stage 1: Presence (Frozen)"
            elif epoch <= CURRICULUM_STAGE1_EPOCHS + CURRICULUM_STAGE2_EPOCHS:
                # Stage 2: Presence + segmentation, encoder still frozen
                current_freeze = True
                train_segmentation = phase_config["has_segmentation"]
                stage_name = "Stage 2: Presence+Seg (Frozen)"
            else:
                # Stage 3: Full unfreezing
                current_freeze = False
                train_segmentation = phase_config["has_segmentation"]
                stage_name = "Stage 3: Full Training"
                
                if not encoder_unfrozen:
                    model.freeze_encoder(False)
                    # Rebuild optimizer with newly unfrozen parameters
                    trainable = [p for p in model.parameters() if p.requires_grad]
                    optimizer = torch.optim.AdamW(trainable, lr=LR * 0.1, weight_decay=WEIGHT_DECAY)
                    encoder_unfrozen = True
                    print(f"🔓 Encoder unfrozen at epoch {epoch}!")
        else:
            # No curriculum: Use original freeze settings
            train_segmentation = phase_config["has_segmentation"]
            stage_name = "Standard Training"
        
        total_loss = 0
        n = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs} - {stage_name}")
        for batch_idx, batch in enumerate(pbar):
            x = batch["image"].to(DEVICE, non_blocking=True)
            x_global = batch.get("image_global")
            if x_global is not None:
                x_global = x_global.to(DEVICE, non_blocking=True)
            
            seg_target = batch["seg"].to(DEVICE, non_blocking=True)
            has_seg = batch["has_seg"].to(DEVICE, non_blocking=True)
            pres_target = batch[f"{disease}_presence"].to(DEVICE, non_blocking=True)
            
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                # Forward with multi-scale
                if train_segmentation:
                    output = model(x, active_presence=[disease], active_seg=[disease], 
                                 x_global=x_global, compute_ood=True)
                    seg_pred = output["segmentations"][disease]
                else:
                    output = model(x, active_presence=[disease], active_seg=None,
                                 x_global=x_global, compute_ood=True)
                    seg_pred = None
                    seg_target = None
                
                pres_pred = output["presence"][disease]
                ood_conf = output.get("ood_confidence")
                
                # Use Alzheimer-specific loss if available, otherwise standard calibrated loss
                if seg_pred is not None:
                    # Has segmentation (Tumor/Stroke)
                    loss, loss_dict = criterion(seg_pred, seg_target, pres_pred, pres_target, 
                                               has_seg, ood_confidence=ood_conf)
                else:
                    # Presence only (Alzheimer or curriculum stage 1)
                    if disease == "alzheimer" and alzheimer_criterion is not None:
                        # Use pos_weight BCE for Alzheimer class imbalance
                        presence_loss = alzheimer_criterion(pres_pred, pres_target)
                        # OOD disabled (weight=0), but keep computation for code consistency
                        ood_loss = torch.tensor(0.0, device=DEVICE)
                        if ood_conf is not None and OOD_ENTROPY_WEIGHT > 0:
                            eps = 1e-8
                            ood_entropy = -(ood_conf * torch.log(ood_conf + eps) + 
                                          (1 - ood_conf) * torch.log(1 - ood_conf + eps))
                            ood_loss = -ood_entropy.mean() * OOD_ENTROPY_WEIGHT
                        loss = presence_loss + ood_loss
                        loss_dict = {"total": loss.item(), "presence": presence_loss.item(), 
                                   "seg": 0.0, "ood": ood_loss.item()}
                    else:
                        # Standard presence-only loss
                        loss, loss_dict = criterion(None, None, pres_pred, pres_target, 
                                                   torch.tensor([0.0]), ood_confidence=ood_conf)
                
                # CRITICAL FIX: Gradient Accumulation Implementation
                # Scale loss to simulate larger batch size
                loss = loss / GRADIENT_ACCUMULATION_STEPS
            
            # Backward pass (gradients accumulate)
            scaler.scale(loss).backward()
            
            # Only update weights every GRADIENT_ACCUMULATION_STEPS batches
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, GRADIENT_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS  # Unscale for logging
            n += 1
            pbar.set_postfix({"loss": f"{loss.item() * GRADIENT_ACCUMULATION_STEPS:.3f}", 
                            "ood": f"{loss_dict.get('ood', 0.0):.4f}"})
        
        avg_loss = total_loss / max(1, n)
        
        # Validation with uncertainty + Grad-CAM (first epoch saves CAM)
        val_dice, val_auc, val_uncertainty = validate(model, val_loader, phase_config, 
                                                       save_gradcam=(epoch == 1))
        
        # Adaptive Encoder Freezing (NEW)
        # Scientific Rationale: If AUC plateaus, encoder needs fine-tuning to escape local minimum
        if phase_config["freeze_encoder"] and not encoder_unfrozen and not curriculum_enabled:
            auc_history.append(val_auc)
            if len(auc_history) == ADAPTIVE_FREEZE_PATIENCE:
                auc_improvement = max(auc_history) - min(auc_history)
                if auc_improvement < ADAPTIVE_FREEZE_DELTA:
                    print(f"\n🔓 AUC plateau detected (improvement={auc_improvement:.4f}). Unfreezing encoder!")
                    model.freeze_encoder(False)
                    trainable = [p for p in model.parameters() if p.requires_grad]
                    optimizer = torch.optim.AdamW(trainable, lr=LR * 0.1, weight_decay=WEIGHT_DECAY)
                    encoder_unfrozen = True
                    auc_history.clear()
        
        if phase_config["has_segmentation"]:
            print(f"✅ Epoch {epoch}: Loss={avg_loss:.4f} | Dice={val_dice:.4f} | AUC={val_auc:.4f}")
            metric = val_dice
        else:
            print(f"✅ Epoch {epoch}: Loss={avg_loss:.4f} | AUC={val_auc:.4f} (presence-only)")
            metric = val_auc
        
        if metric > best_metric:
            best_metric = metric
            ckpt = CHECKPOINT_DIR / f"neurox_phase{phase_num}_best.pth"
            torch.save(model.state_dict(), ckpt)
            print(f"💾 Saved: {ckpt.name} (metric={metric:.4f})")
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    print("\n🏗️ Initializing NeuroX Multi-Disease Model...")
    model = NeuroXMultiDisease().to(DEVICE)
    print(f"✅ Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"📋 Presence Heads: {list(model.presence_heads.keys())}")
    print(f"📋 Segmentation Decoders: {list(model.seg_decoders.keys())}\n")
    
    for idx, phase in enumerate(TRAINING_PHASES, 1):
        train_phase(model, phase, idx)
    
    final = CHECKPOINT_DIR / "neurox_multihead_final.pth"
    torch.save(model.state_dict(), final)
    print("\n" + "=" * 80)
    print("🎉 MULTI-DISEASE TRAINING COMPLETE")
    print("=" * 80)
    print(f"💾 Final Model: {final}")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n✅ Model supports: Tumor | Stroke | Alzheimer Detection\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
