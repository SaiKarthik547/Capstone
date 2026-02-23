
import os
import sys
import random
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, accuracy_score,
    roc_curve
)
from collections import Counter
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════
#  0. GLOBAL SAFETY CHECKS
# ═══════════════════════════════════════════════════════════════════════════
print("="*80)
print("🔍 GLOBAL SAFETY CHECKS")
print(f"CUDA Available: {torch.cuda.is_available()}")
device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print(f"Device: {device_name}")
print("="*80)

# ═══════════════════════════════════════════════════════════════════════════
#  1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROI_SIZE = (96, 96, 96)
BATCH_SIZE = 1
NUM_WORKERS = 2         
WEIGHT_DECAY = 1e-5
EPOCHS = 80          # Optimized 80-epoch curriculum
USE_AMP = True       # AMP enabled for memory efficiency and speed

# Multi-task loss weights (Final Balanced Setting)
LAMBDA_TUMOR  = 1.5
LAMBDA_STROKE = 1.5
LAMBDA_CLS    = 2.0

CHECKPOINT_DIR = Path("./checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Dynamic weights will be set in main()
STROKE_POS_WEIGHT = None 
ALZ_POS_WEIGHT = None

# Cache dir for resized volumes
CACHE_DIR = Path("/kaggle/working/cache")
CACHE_DIR.mkdir(exist_ok=True, parents=True)

# DEBUG MODE
DEBUG = False  # Set True for quick pipeline check (20 samples, 3 epochs)

if DEBUG:
    print("🔬 DEBUG MODE ENABLED: Training restricted to 20 samples/dataset, 3 epochs.")
    EPOCHS = 3

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ═══════════════════════════════════════════════════════════════════════════
#  2. PREPROCESSING (UNIFIED & DETERMINISTIC)
# ═══════════════════════════════════════════════════════════════════════════

def load_nifti(path: Path) -> np.ndarray:
    try:
        img = nib.load(str(path))
        data = img.get_fdata()
        return np.asarray(data, dtype=np.float32)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return np.zeros(ROI_SIZE, dtype=np.float32)

def preprocess_volume(volume: np.ndarray, is_mask: bool = False) -> torch.Tensor:
    """Unified deterministic preprocessing."""
    volume = volume.astype(np.float32)
    
    if not is_mask:
        mean = volume.mean()
        std = volume.std() + 1e-8
        volume = (volume - mean) / std
    
    volume = torch.from_numpy(volume)
    if volume.ndim == 3:
        volume = volume.unsqueeze(0).unsqueeze(0)
    elif volume.ndim == 4:
        volume = volume.unsqueeze(0)
        
    volume = F.interpolate(
        volume,
        size=ROI_SIZE,
        mode="nearest" if is_mask else "trilinear",
        align_corners=None if is_mask else False
    )
    
    return volume.squeeze(0)  # [C=1, D, H, W]

# ═══════════════════════════════════════════════════════════════════════════
#  3. DATASETS
# ═══════════════════════════════════════════════════════════════════════════

class TumorDataset(Dataset):
    """BraTS 2020: T1ce Input -> [ET, NCR, ED] Independent Binary Masks
    
    Using mutually exclusive labels on ROI:
    - ET  = Enhancing Tumor (label 4)
    - NCR = Necrotic Core (label 1)
    - ED  = Edema (label 2)
    """
    def __init__(self, root_path: str, debug=False):
        self.root = Path(root_path)
        self.cases = self._find_cases()
        if debug:
            self.cases = self.cases[:20]
            print(f"⚠️ DEBUG: Tumor dataset restricted to {len(self.cases)} cases")
        print(f"✅ Tumor Dataset: {len(self.cases)} cases")

    def _find_cases(self):
        cases = []
        for case_dir in self.root.rglob("BraTS20_Training_*"):
            if not case_dir.is_dir(): continue
            t1ce = list(case_dir.glob("*_t1ce.nii*"))
            seg = list(case_dir.glob("*_seg.nii*"))
            if t1ce and seg:
                cases.append({"t1ce": t1ce[0], "seg": seg[0]})
        return cases
    
    def __len__(self): return len(self.cases)
    
    def __getitem__(self, idx):
        case = self.cases[idx]
        cache_key = CACHE_DIR / f"tumor_{idx}.npz"
        
        # STEP 7: Load from cache if available
        if cache_key.exists():
            data = np.load(cache_key)
            image = torch.from_numpy(data["image"])
            target = torch.from_numpy(data["target"])
        else:
            img = load_nifti(case["t1ce"])
            image = preprocess_volume(img, is_mask=False)
            
            seg_vol = load_nifti(case["seg"])
            seg = preprocess_volume(seg_vol, is_mask=True)
            seg = seg.round()
            
            # Independent binary masks (mutually exclusive)
            # BraTS Labels: 1=NCR/NET, 2=ED, 4=ET
            seg_et  = (seg == 4.0).float()   
            seg_ncr = (seg == 1.0).float()   
            seg_ed  = (seg == 2.0).float()   
            
            target = torch.cat([seg_et, seg_ncr, seg_ed], dim=0)  # [3, 96, 96, 96]
            
            np.savez_compressed(cache_key, image=image.numpy(), target=target.numpy())
        
        has_tumor = 1.0 if target.sum() > 0 else 0.0
        
        return {
            "image": image,
            "seg": target,
            "has_seg": torch.tensor([1.0]),
            "presence": {
                "tumor": torch.tensor([has_tumor]),
                "stroke": torch.tensor([0.0]),
                "alzheimer": torch.tensor([0.0])
            }
        }

class StrokeDataset(Dataset):
    """ISLES 2022: DWI/ADC Input -> Binary Mask Target"""
    def __init__(self, root_path: str, debug=False):
        self.root = Path(root_path)
        self.cases = self._find_cases()
        if debug:
            self.cases = self.cases[:20]
            print(f"⚠️ DEBUG: Stroke dataset restricted to {len(self.cases)} cases")
        print(f"✅ Stroke Dataset: {len(self.cases)} cases")

    def _find_cases(self):
        cases = []
        base = self.root / "ISLES-2022" / "ISLES-2022"
        for sub_dir in base.glob("sub-strokecase*"):
            dwi_dir = sub_dir / "ses-0001" / "dwi"
            if not dwi_dir.exists(): continue
            
            # Recursive search, filter empty files
            dwi_candidates = list(dwi_dir.rglob("*.nii*"))
            valid_dwi = [f for f in dwi_candidates if f.is_file() and f.stat().st_size > 1024]
            
            if not valid_dwi:
                continue

            msk_dir = base / "derivatives" / sub_dir.name / "ses-0001"
            if not msk_dir.exists(): continue
            
            msk_candidates = list(msk_dir.rglob("*.nii*"))
            valid_msk = [f for f in msk_candidates if f.is_file() and "msk" in f.name and f.stat().st_size > 1024]
            
            if valid_dwi and valid_msk:
                # Prefer ADC file
                dwi_path = next((f for f in valid_dwi if "adc" in f.name.lower()), None)
                if dwi_path is None:
                    dwi_path = max(valid_dwi, key=lambda x: x.stat().st_size)
                cases.append({"dwi": dwi_path, "msk": valid_msk[0]})
        return cases

    def __len__(self): return len(self.cases)

    def __getitem__(self, idx):
        case = self.cases[idx]
        cache_key = CACHE_DIR / f"stroke_{idx}.npz"
        
        if cache_key.exists():
            data = np.load(cache_key)
            image = torch.from_numpy(data["image"])
            target = torch.from_numpy(data["target"])
        else:
            vol_img = load_nifti(case["dwi"])
            vol_msk = load_nifti(case["msk"])
            
            # Strict alignment check
            assert vol_img.shape == vol_msk.shape, f"Stroke shape mismatch: {vol_img.shape} vs {vol_msk.shape}"

            image = preprocess_volume(vol_img, is_mask=False)
            mask = preprocess_volume(vol_msk, is_mask=True)
            target = (mask > 0).float()
            
            np.savez_compressed(cache_key, image=image.numpy(), target=target.numpy())
        
        has_stroke = 1.0 if target.sum() > 0 else 0.0
        
        return {
            "image": image,
            "seg": target,
            "has_seg": torch.tensor([1.0]),
            "presence": {
                "tumor": torch.tensor([0.0]),
                "stroke": torch.tensor([has_stroke]),
                "alzheimer": torch.tensor([0.0])
            }
        }

class AlzheimerDataset(Dataset):
    def __init__(self, records, augment=False):
        self.records = records
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        vol = load_nifti(record["path"])
        image = preprocess_volume(vol, is_mask=False)

        if self.augment:
            if random.random() < 0.5:
                image = torch.flip(image, [2])
            if random.random() < 0.5:
                image = torch.flip(image, [3])

        return {
            "image": image,
            "seg": torch.zeros((1, *ROI_SIZE)),
            "has_seg": torch.tensor([0.0]),
            "presence": {
                "tumor": torch.tensor([0.0]),
                "stroke": torch.tensor([0.0]),
                "alzheimer": torch.tensor([float(record["label"])])
            }
        }

def build_alzheimer_subject_split(data_root, seed=42, debug=False):
    """
    Build AD vs CN dataset from ADNI-style folder structure.
    Performs SUBJECT-LEVEL stratified split.
    """

    CN_ROOT = Path(data_root) / "ecc" / "ADNI"
    AD_ROOT = Path(data_root) / "abb" / "ADNI"

    samples = []

    def collect(root, label):
        if not root.exists():
            print(f"⚠️ Warning: Alzheimer root {root} does not exist!")
            return
        for subject in os.listdir(root):
            subject_path = root / subject
            if not subject_path.is_dir():
                continue
            for path, _, files in os.walk(subject_path):
                for f in files:
                    # Scan for both .nii and .nii.gz for robustness
                    if f.endswith(".nii") or f.endswith(".nii.gz"):
                        samples.append({
                            "subject": subject,
                            "path": str(Path(path) / f),
                            "label": label
                        })

    collect(CN_ROOT, 0)
    collect(AD_ROOT, 1)

    if not samples:
        print("❌ CRITICAL: No Alzheimer samples found! Check data_root paths.")
        return [], []

    df = pd.DataFrame(samples)

    if debug:
        # Reduce dataset for debug while keeping class balance approx
        df = df.sample(n=min(200, len(df)), random_state=seed)

    # Subject-level grouping for stratification
    subject_df = df.groupby("subject")["label"].first().reset_index()

    train_subj, val_subj = train_test_split(
        subject_df,
        test_size=0.2,
        stratify=subject_df["label"],
        random_state=seed
    )

    train_df = df[df["subject"].isin(train_subj["subject"])]
    val_df   = df[df["subject"].isin(val_subj["subject"])]

    print("✅ Alzheimer Subject-Level Split:")
    print("   Train subjects:", len(train_subj))
    print("   Val subjects:", len(val_subj))
    print("   Train scans:", len(train_df))
    print("   Val scans:", len(val_df))

    return train_df.to_dict("records"), val_df.to_dict("records")

# ═══════════════════════════════════════════════════════════════════════════
#  4. MODEL ARCHITECTURE
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
        # 3D Transformer bottleneck
        self.bottleneck = TransformerBottleneck3D(128, 4, 8, 256, 0.2)
    
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
        # Handle 2-channel input gracefully (safety)
        if x.shape[1] == 2:
            x = x.mean(dim=1, keepdim=True)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        return {"enc1": e1, "enc2": e2, "enc3": e3, "bottleneck": b}


class PresenceHead(nn.Module):
    """Binary presence detector with MC Dropout uncertainty."""
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
    """Multitask model with selective forward.

    Selective forward: only activate heads needed per batch.
    Alzheimer uses a dedicated AlzheimerEncoder that receives raw MRI directly.
    SharedEncoder is used only for Tumor / Stroke.
    """
    def __init__(self):
        super().__init__()
        self.encoder = SharedEncoder(in_channels=1)
        # Tumor & Stroke presence: transformer bottleneck -> PresenceHead
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
        # Raw MRI -> independent 3D CNN -> dual pool -> MLP -> AD logit
        # No shared features with SharedEncoder.
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
                    # Tumor / Stroke: bottleneck -> PresenceHead (unchanged)
                    presence[key] = self.presence_heads[key](features["bottleneck"])
        segmentations = {}
        if active_seg:
            for key in active_seg:
                if key in self.seg_decoders:
                    segmentations[key] = self.seg_decoders[key](features)
        return {"presence": presence, "segmentations": segmentations}

# ═══════════════════════════════════════════════════════════════════════════
#  5. TRAINING UTILS (LOSS & METRICS)
# ═══════════════════════════════════════════════════════════════════════════

def compute_dice_loss(logits, targets):
    # Channel-Wise Dice for medical imaging
    # smooth=1.0 is standard for BraTS/ISLES to stabilize denominator at batch_size=1
    smooth = 1.0
    probs = torch.sigmoid(logits)
    
    # intersection & union per channel/batch item
    dims = (2, 3, 4)
    intersection = (probs * targets).sum(dim=dims)
    union = probs.sum(dim=dims) + targets.sum(dim=dims)
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()

def quick_dice(logits, targets):
    """Dice Score (0-1) for logging. No gradients."""
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    
    inter = (preds * targets).sum(dim=(2, 3, 4))
    union = preds.sum(dim=(2, 3, 4)) + targets.sum(dim=(2, 3, 4))
    
    dice = (2. * inter) / (union + 1e-6)
    return dice.mean().item()

def focal_loss(logits, targets, gamma=2.0, alpha=0.25):
    """Focal loss: focuses training on hard, misclassified voxels.
    
    Replaces BCE in segmentation. alpha balances foreground/background.
    gamma=2.0 conservative start — increase later if needed.
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = torch.exp(-bce)
    focal = alpha * (1 - p_t) ** gamma * bce
    return focal.mean()

def train_step(model, batch, task, optimizer, scaler, epoch, skip_backward=False):
    """
    Standard training step for a single task.
    If skip_backward=True, returns the total loss tensor without performing backward or step.
    """
    img = batch["image"].to(DEVICE)
    tgt = batch["seg"].to(DEVICE) if task in ["tumor", "stroke"] else None
    
    # ─── Forward ──────────────────────────────────────────────────────────
    with torch.amp.autocast("cuda", enabled=USE_AMP):
        if task == "alzheimer":
            out = model(img, active_presence=["alzheimer"])["presence"]["alzheimer"]
            if ALZ_POS_WEIGHT is not None:
                loss = F.binary_cross_entropy_with_logits(out, batch["presence"]["alzheimer"].to(DEVICE), pos_weight=ALZ_POS_WEIGHT.to(DEVICE))
            else:
                loss = F.binary_cross_entropy_with_logits(out, batch["presence"]["alzheimer"].to(DEVICE))
            total_loss = loss
        
        elif task in ["tumor", "stroke"]:
            res = model(img, active_presence=[task], active_seg=[task])
            pres_logits = res["presence"][task]
            seg_logits  = res["segmentations"][task]
            
            # Loss A: Presence (Controlled by global LAMBDA_CLS)
            loss_pres = F.binary_cross_entropy_with_logits(pres_logits, batch["presence"][task].to(DEVICE))
            
            # Loss B: Segmentation (0.8 Dice + 0.2 BCE)
            loss_dice = compute_dice_loss(seg_logits, tgt)
            
            if task == "stroke":
                # Use dynamic STROKE_POS_WEIGHT (capped at 50)
                loss_bce = F.binary_cross_entropy_with_logits(seg_logits, tgt, pos_weight=STROKE_POS_WEIGHT.to(DEVICE))
            else:
                loss_bce = F.binary_cross_entropy_with_logits(seg_logits, tgt)
                
            loss_seg = 0.8 * loss_dice + 0.2 * loss_bce
            
            lam = LAMBDA_TUMOR if task == "tumor" else LAMBDA_STROKE
            total_loss = lam * loss_seg + LAMBDA_CLS * loss_pres

    # ─── Stability Check ──────────────────────────────────────────────────
    if not torch.isfinite(total_loss):
        print(f"⚠️ Non-finite loss detected in {task} at Epoch {epoch}. Skipping.")
        return None

    # Logit Quality Audit (Early Warning)
    with torch.no_grad():
        if task in ["tumor", "stroke"]:
            std = seg_logits.std().item()
            if std > 10.0:
                print(f"⚠️ High Logit STD ({std:.2f}) in {task} at Epoch {epoch}")

    # ─── Backward & Step ──────────────────────────────────────────────────
    if skip_backward:
        return total_loss

    # Redundant zero_grad removed - handled in main loop
    scaler.scale(total_loss).backward()
    
    # Gradient Clipping (1.0)
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    scaler.step(optimizer)
    scaler.update()
    
    return total_loss.item()

def evaluate_alzheimer_full(model, loader):
    """Full Alzheimer classification metrics: AUC, Accuracy, Precision, Recall, F1.

    Uses dedicated AlzheimerEncoder — self-contained forward pass on raw MRI.
    Correctly accesses batch["presence"]["alzheimer"] per AlzheimerDataset format.
    """
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(DEVICE)
            lbl = batch["presence"]["alzheimer"].float()  # shape [B, 1]
            # AlzheimerEncoder is self-contained: raw MRI -> logit
            logit = model.alz_encoder(img)                # [B, 1]
            all_logits.append(logit.cpu())
            all_labels.append(lbl)
    model.train()

    logits  = torch.cat(all_logits).view(-1).numpy()
    labels  = torch.cat(all_labels).view(-1).numpy()
    probs   = 1.0 / (1.0 + np.exp(-logits))

    logit_mean = float(logits.mean())
    logit_std  = float(logits.std())
    tqdm.write(
        f"   📊 Logit Diagnostics | mean={logit_mean:.4f}  std={logit_std:.4f}"
        f"  {'✅ OK' if logit_std > 0.3 else '⚠️ LOW STD — encoder may be dormant'}"
    )

    try:
        auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5
    except Exception:
        auc = 0.5

    # Fixed-threshold metrics (threshold=0.5)
    preds_fixed = (probs >= 0.5).astype(int)
    acc_fixed   = accuracy_score(labels, preds_fixed)
    prec_fixed  = precision_score(labels, preds_fixed, zero_division=0)
    rec_fixed   = recall_score(labels, preds_fixed, zero_division=0)
    f1_fixed    = f1_score(labels, preds_fixed, zero_division=0)

    # Optimal-threshold metrics calculation
    opt_thresh = 0.5
    if len(np.unique(labels)) > 1:
        try:
            fpr, tpr, thresholds = roc_curve(labels, probs)
            optimal_idx = (tpr - fpr).argmax()
            opt_thresh  = float(thresholds[optimal_idx])
        except Exception:
            pass
    preds_opt = (probs >= opt_thresh).astype(int)
    acc_opt   = accuracy_score(labels, preds_opt)
    prec_opt  = precision_score(labels, preds_opt, zero_division=0)
    rec_opt   = recall_score(labels, preds_opt, zero_division=0)
    f1_opt    = f1_score(labels, preds_opt, zero_division=0)

    tqdm.write(
        f"   🛡️  Alzheimer Val  | AUC={auc:.4f}\n"
        f"   📌 Fixed  thr=0.50 | Acc={acc_fixed:.3f}  Prec={prec_fixed:.3f}  "
        f"Rec={rec_fixed:.3f}  F1={f1_fixed:.3f}\n"
        f"   🎯 Optimal thr={opt_thresh:.2f} | Acc={acc_opt:.3f}  Prec={prec_opt:.3f}  "
        f"Rec={rec_opt:.3f}  F1={f1_opt:.3f}"
    )

    # Return fixed-threshold metrics for history (consistent across epochs)
    acc, prec, rec, f1 = acc_fixed, prec_fixed, rec_fixed, f1_fixed
    return {"auc": auc, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

def check_batches(tumor_loader, stroke_loader):
    """Pre-training batch sanity check."""
    print("\n🔍 PRE-TRAINING BATCH CHECK 🔍")
    try:
        batch = next(iter(tumor_loader))
        img, seg = batch["image"], batch["seg"]
        print(f"✅ Tumor Batch: Img {img.shape} Range [{img.min():.2f}, {img.max():.2f}] | Seg {seg.shape} Sum {seg.sum().item():.0f}")
        if seg.sum() < 1:
            print("⚠️ WARNING: Tumor mask sum is 0!")

        batch = next(iter(stroke_loader))
        img, seg = batch["image"], batch["seg"]
        print(f"✅ Stroke Batch: Img {img.shape} Range [{img.min():.2f}, {img.max():.2f}] | Seg {seg.shape} Sum {seg.sum().item():.0f}")
        if seg.sum() < 1:
            print("⚠️ WARNING: Stroke mask sum is 0 (Small lesion lost in resize?)")
    except Exception as e:
        print(f"❌ Batch Check Failed: {e}")
    print("="*40 + "\n")

# ═══════════════════════════════════════════════════════════════════════════
#  6. MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════

def main():
    global USE_AMP
    # 1. Prepare Datasets
    tumor_ds = TumorDataset("/kaggle/input/datasets/awsaf49/brats20-dataset-training-validation/", debug=DEBUG)
    stroke_ds = StrokeDataset("/kaggle/input/datasets/orvile/isles-2022-brain-stoke-dataset/", debug=DEBUG)
    
    alz_train_data, alz_val_data = build_alzheimer_subject_split(
        "/kaggle/input/datasets/muhammadzahraan/3d-mri-scans-for-alzheimer-disease",
        seed=SEED,
        debug=DEBUG
    )
    # New AlzheimerDataset supports record-based initialization
    alz_train_ds = AlzheimerDataset(alz_train_data, augment=True)
    alz_val_ds = AlzheimerDataset(alz_val_data, augment=False)

    # Compute Alzheimer class weight from training labels
    global ALZ_POS_WEIGHT
    if alz_train_data:
        alz_labels = [record["label"] for record in alz_train_data]
        counts = Counter(alz_labels)
        pos_w = counts.get(0, 1) / max(counts.get(1, 1), 1)  # N_neg / N_pos
        ALZ_POS_WEIGHT = torch.tensor([pos_w], device=DEVICE)
        print(f"✅ ALZ_POS_WEIGHT = {pos_w:.2f} (neg={counts.get(0,0)}, pos={counts.get(1,0)})")
    else:
        ALZ_POS_WEIGHT = None
    
    # 2. DataLoaders
    loader_kwargs = dict(
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0)
    )
    tumor_loader  = DataLoader(tumor_ds, **loader_kwargs)
    stroke_loader = DataLoader(stroke_ds, **loader_kwargs)
    alz_train_loader = DataLoader(alz_train_ds, **loader_kwargs)
    alz_val_loader   = DataLoader(alz_val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=NUM_WORKERS, pin_memory=True,
                                  persistent_workers=(NUM_WORKERS > 0))
    # Calculate Stroke Class Imbalance (Dynamic Once)
    print("\n⚖️  Calculating Stroke Class Imbalance...")
    total_pos = 0
    total_voxels = 0
    for i in range(len(stroke_ds)):
        sample = stroke_ds[i]
        mask = sample["seg"]
        total_pos += mask.sum().item()
        total_voxels += mask.numel()
    
    total_neg = total_voxels - total_pos
    s_weight = min(total_neg / (total_pos + 1e-6), 50.0)
    global STROKE_POS_WEIGHT
    STROKE_POS_WEIGHT = torch.tensor([s_weight], device=DEVICE)
    print(f"   Stroke pos_weight: {s_weight:.2f} (Capped at 50)")

    # ═══════════════════════════════════════════════════════════════════════════
    #  7. OPTIMIZER & SCALER (PHASE-AWARE GROUPS)
    # ═══════════════════════════════════════════════════════════════════════════

    # 3. Setup Model & Optimizers
    model = NeuroXMultiDisease().to(DEVICE)

    # Separate Conv vs Transformer parameters for shared encoder components
    conv_params = []
    transformer_params = []

    for name, p in model.named_parameters():
        # Includes shared encoder convs, seg decoders, and presence heads
        if "alz_encoder" in name:
            continue
        if "bottleneck" in name:
            transformer_params.append(p)
        else:
            conv_params.append(p)

    optimizer_shared = torch.optim.AdamW([
        {"params": conv_params,        "lr": 1e-4}, 
        {"params": transformer_params, "lr": 5e-5}
    ], weight_decay=WEIGHT_DECAY)

    optimizer_alz = torch.optim.AdamW(model.alz_encoder.parameters(), lr=1e-4, weight_decay=WEIGHT_DECAY)

    # Initial scalers
    scaler_shared = torch.amp.GradScaler(enabled=USE_AMP)
    scaler_alz    = torch.amp.GradScaler(enabled=USE_AMP)

    # Dual Schedulers (Removed for stability - using constant LRs)
    scheduler_shared = None
    scheduler_alz    = None
    
    # Pre-Training Check
    check_batches(tumor_loader, stroke_loader)
    
    print(f"🚀 Starting Training (DEBUG={DEBUG}, EPOCHS={EPOCHS})...")
    
    # Metrics history — defined BEFORE loop so each epoch appends cumulatively
    metrics_history = {
        "epoch":         [],
        "tumor_et":      [],
        "tumor_ncr":     [],
        "tumor_ed":      [],
        "tumor_mean":    [],
        "stroke_dice":   [],
        "alz_auc":       [],
        "alz_accuracy":  [],
        "alz_precision": [],
        "alz_recall":    [],
        "alz_f1":        [],
    }
    
    for epoch in range(1, EPOCHS + 1):
        global LAMBDA_CLS
        # ─── 80-EPOCH CURRICULUM ──────────────────────────────────
        if epoch <= 20:
            phase_name = " PHASE 1: ALZHEIMER PRETRAINING"
            TRAIN_ALZ, TRAIN_SEG = True, False
        else:
            phase_name = " PHASE 2: SEGMENTATION (STABILIZED)"
            TRAIN_ALZ, TRAIN_SEG = False, True

        # ─── PHASE TRANSITIONS (CURRICULUM LOGIC) ──────────────────────────
        
        if epoch == 1:
            print(f"\n❄️  PHASE 1: ALZHEIMER PRE-TRAIN")
            for name, p in model.named_parameters():
                if "alz_encoder" in name: p.requires_grad = True
                else: p.requires_grad = False
            USE_AMP = True
            TRAIN_ALZ, TRAIN_SEG = True, False
            LAMBDA_CLS = 2.0

        if epoch == 21:
            print(f"\n🔥 PHASE 2A: SEGMENTATION WARMUP (AMP OFF, TRANS FROZEN)")
            for name, p in model.named_parameters():
                if "alz_encoder" in name: p.requires_grad = False
                elif "bottleneck" in name: p.requires_grad = False
                else: p.requires_grad = True
            
            USE_AMP = False
            LAMBDA_CLS = 0.0 # Pure segmentation phase
            scaler_shared = torch.amp.GradScaler(enabled=USE_AMP)
            
            # Dampening LR for Phase Transition (Epoch 21-22)
            optimizer_shared.param_groups[0]['lr'] = 5e-5 # convs
            TRAIN_ALZ, TRAIN_SEG = False, True

        if epoch == 23:
            print(f"\n📈 Phase 2A': Standardizing Seg LR")
            optimizer_shared.param_groups[0]['lr'] = 1e-4

        if epoch == 26:
            print(f"\n🚀 PHASE 2B: FULL SEGMENTATION (AMP ON, TRANS UNFROZEN)")
            for name, p in model.named_parameters():
                if "bottleneck" in name: p.requires_grad = True
            
            USE_AMP = True
            scaler_shared = torch.amp.GradScaler(enabled=USE_AMP)
            optimizer_shared.param_groups[1]['lr'] = 5e-5 # transformer

        # ─── Loop Setup ─────────────────────────────────────────────────────
        model.train()
        
        tumor_iter  = iter(tumor_loader)  if TRAIN_SEG else None
        stroke_iter = iter(stroke_loader) if TRAIN_SEG else None
        alz_iter    = iter(alz_train_loader) if TRAIN_ALZ else None
        
        t_losses, s_losses, a_losses = [], [], []
        t_et_dices, t_ncr_dices, t_ed_dices = [], [], []
        s_dices = []
        
        avg_et, avg_ncr, avg_ed, avg_tumor_mean, avg_stroke, avg_alz_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        if TRAIN_ALZ:
            max_steps = len(alz_train_loader)
        elif TRAIN_SEG:
            max_steps = min(len(tumor_loader), len(stroke_loader))
        else:
            max_steps = 0

        print(f"\n{'='*75}")
        print(f"  EPOCH {epoch}/{EPOCHS} | TRAIN_ALZ={TRAIN_ALZ} TRAIN_SEG={TRAIN_SEG} AMP={USE_AMP}")
        print(f"{'='*75}")
        
        for step in range(max_steps):
            if TRAIN_SEG: optimizer_shared.zero_grad()
            if TRAIN_ALZ: optimizer_alz.zero_grad()
            
            # --- Alzheimer Update (Phase 1) ---
            if TRAIN_ALZ:
                try:
                    batch = next(alz_iter)
                    loss = train_step(model, batch, "alzheimer", optimizer_alz, scaler_alz, epoch)
                    if loss is not None: a_losses.append(loss)
                except (StopIteration, TypeError):
                    pass
            
            # --- Unified Segmentation Update (Phase 2) ---
            if TRAIN_SEG:
                try:
                    t_batch = next(tumor_iter)
                    s_batch = next(stroke_iter)
                    
                    # Outer autocast removed (handled in train_step)
                    loss_tumor  = train_step(model, t_batch, "tumor", optimizer_shared, scaler_shared, epoch, skip_backward=True)
                    loss_stroke = train_step(model, s_batch, "stroke", optimizer_shared, scaler_shared, epoch, skip_backward=True)
                    
                    if loss_tumor is not None and loss_stroke is not None:
                        total_seg_loss = loss_tumor + loss_stroke
                        scaler_shared.scale(total_seg_loss).backward()
                        
                        scaler_shared.unscale_(optimizer_shared)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        
                        scaler_shared.step(optimizer_shared)
                        scaler_shared.update()
                        
                        t_losses.append(loss_tumor.item())
                        s_losses.append(loss_stroke.item())
                        
                        # --- Metrics Logging (CRITICAL Fix) ---
                        with torch.no_grad():
                            # Mode flips removed (already in model.train())
                            
                            # Tumor Metrics
                            t_res = model(t_batch["image"].to(DEVICE), active_seg=["tumor"])
                            t_out = t_res["segmentations"]["tumor"]
                            et_d  = quick_dice(t_out[:, 0:1], t_batch["seg"][:, 0:1].to(DEVICE))
                            ncr_d = quick_dice(t_out[:, 1:2], t_batch["seg"][:, 1:2].to(DEVICE))
                            ed_d  = quick_dice(t_out[:, 2:3], t_batch["seg"][:, 2:3].to(DEVICE))
                            t_et_dices.append(et_d)
                            t_ncr_dices.append(ncr_d)
                            t_ed_dices.append(ed_d)
                            
                            # Stroke Metrics
                            s_res = model(s_batch["image"].to(DEVICE), active_seg=["stroke"])
                            s_out = s_res["segmentations"]["stroke"]
                            s_d = quick_dice(s_out, s_batch["seg"].to(DEVICE))
                            s_dices.append(s_d)

                        # First 10 steps startup debug
                        if epoch == 21 and step < 10:
                            p_mean = torch.sigmoid(s_out).mean().item()
                            g_mean = s_batch["seg"].mean().item()
                            tqdm.write(f"   [P2 Startup Step {step}] Stroke Pred Mean: {p_mean:.4f} | GT Mean: {g_mean:.4f}")

                except (StopIteration, TypeError):
                    pass
            
            # Print progress every 50 steps
            if (step + 1) % 50 == 0 or step == max_steps - 1:
                log_parts = [f"  Step {step+1:>4}/{max_steps}"]
                if TRAIN_SEG:
                    t_l = np.mean(t_losses[-50:]) if t_losses else 0.0
                    s_l = np.mean(s_losses[-50:]) if s_losses else 0.0
                    et_d = np.mean(t_et_dices[-50:]) if t_et_dices else 0.0
                    s_d  = np.mean(s_dices[-50:]) if s_dices else 0.0
                    log_parts.append(f"T_L={t_l:.3f} | S_L={s_l:.3f} | ET_D={et_d:.3f} | Str_D={s_d:.3f}")
                if TRAIN_ALZ:
                    log_parts.append(f"A_L={np.mean(a_losses[-50:]):.3f}")
                tqdm.write(" | ".join(log_parts))
        
        # ── Epoch Summary ────────────────────────────────────────────────────
        if TRAIN_SEG:
            avg_et  = np.mean(t_et_dices)  if t_et_dices  else 0.0
            avg_ncr = np.mean(t_ncr_dices) if t_ncr_dices else 0.0
            avg_ed  = np.mean(t_ed_dices)  if t_ed_dices  else 0.0
            avg_tumor_mean = (avg_et + avg_ncr + avg_ed) / 3.0
            avg_stroke = np.mean(s_dices) if s_dices else 0.0

            print(f"  🧠 Tumor  | Loss: {np.mean(t_losses) if t_losses else 0:.4f}")
            print(f"     ET  Dice : {avg_et:.4f}  ← primary benchmark")
            print(f"     NCR Dice : {avg_ncr:.4f}")
            print(f"     ED  Dice : {avg_ed:.4f}")
            print(f"     Mean     : {avg_tumor_mean:.4f}")
            print(f"  🩸 Stroke | Loss: {np.mean(s_losses) if s_losses else 0:.4f} | Dice: {avg_stroke:.4f}")
        
        if TRAIN_ALZ:
            avg_alz_loss = np.mean(a_losses) if a_losses else 0.0
            print(f"  🧬 Alz    | Loss: {avg_alz_loss:.4f}")
        sys.stdout.flush()

        if TRAIN_ALZ:
            # Alzheimer Validation — full metrics (AUC + Acc + Prec + Rec + F1)
            alz_metrics = evaluate_alzheimer_full(model, alz_val_loader)
            alz_auc = alz_metrics["auc"]

            # Append metrics
            metrics_history["alz_auc"].append(round(alz_auc, 4))
            metrics_history["alz_accuracy"].append(round(alz_metrics["accuracy"], 4))
            metrics_history["alz_precision"].append(round(alz_metrics["precision"], 4))
            metrics_history["alz_recall"].append(round(alz_metrics["recall"], 4))
            metrics_history["alz_f1"].append(round(alz_metrics["f1"], 4))
        else:
            # Pad Alzheimer metrics with last known value
            last_auc = metrics_history["alz_auc"][-1] if metrics_history["alz_auc"] else 0.0
            metrics_history["alz_auc"].append(last_auc)
            metrics_history["alz_accuracy"].append(metrics_history["alz_accuracy"][-1] if metrics_history["alz_accuracy"] else 0.0)
            metrics_history["alz_precision"].append(metrics_history["alz_precision"][-1] if metrics_history["alz_precision"] else 0.0)
            metrics_history["alz_recall"].append(metrics_history["alz_recall"][-1] if metrics_history["alz_recall"] else 0.0)
            metrics_history["alz_f1"].append(metrics_history["alz_f1"][-1] if metrics_history["alz_f1"] else 0.0)

        # Unified history updates for segmentation
        metrics_history["epoch"].append(epoch)
        metrics_history["tumor_et"].append(round(avg_et, 4))
        metrics_history["tumor_ncr"].append(round(avg_ncr, 4))
        metrics_history["tumor_ed"].append(round(avg_ed, 4))
        metrics_history["tumor_mean"].append(round(avg_tumor_mean, 4))
        metrics_history["stroke_dice"].append(round(avg_stroke, 4))

        # ─── Scheduler Step (Removed for stability) ────────────
        print(f"     LR Shared: {optimizer_shared.param_groups[0]['lr']:.2e} | LR Alz: {optimizer_alz.param_groups[0]['lr']:.2e}")
        
        sys.stdout.flush()
        
        # Checkpoint — Full (for training resume)
        torch.save({
            "model":    model.state_dict(),
            "optimizer_shared": optimizer_shared.state_dict(),
            "optimizer_alz":    optimizer_alz.state_dict(),
            "epoch":    epoch,
            "metrics":  metrics_history,
        }, CHECKPOINT_DIR / "neurox_checkpoint.pth")
        
        # Checkpoint — Inference weights + metrics
        torch.save({
            "model_state": model.state_dict(),
            "metrics":     metrics_history,
        }, CHECKPOINT_DIR / "neurox_model.pth")
        
        print(f"✅ Epoch {epoch} Complete. Checkpoints saved.")

if __name__ == "__main__":
    main()
