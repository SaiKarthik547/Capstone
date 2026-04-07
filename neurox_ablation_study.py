import os
import sys
import random
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    precision_score, recall_score, f1_score, accuracy_score
)
from collections import Counter, defaultdict
import pandas as pd
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
import json
import shutil
import hashlib as _hashlib
from torch.optim.lr_scheduler import CosineAnnealingLR

# ═══════════════════════════════════════════════════════════════════════════
#  0. ABLATION CONFIGURATION (Mission Control)
# ═══════════════════════════════════════════════════════════════════════════

ABLATION_CONFIG = {
    "USE_TRANSFORMER": True,      # 3D Transformer Bottleneck
    "USE_ATTENTION_GATES": True,  # Spatial Attention Gates
    "USE_UNCERTAINTY": True,      # Heteroscedastic Uncertainty Weighting
    "USE_ALZ_ISOLATION": True,    # Dedicated Encoder for Alzheimer's
    "USE_SE_BLOCK": True,         # Squeeze-and-Excitation (Final Layer)
}

# ═══════════════════════════════════════════════════════════════════════════
#  1. GLOBAL CONFIGURATION (Production Grade)
# ═══════════════════════════════════════════════════════════════════════════

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROI_SIZE = (96, 96, 96)
BATCH_SIZE = 1
ACCUM_STEPS = 4
NUM_WORKERS = 1
WEIGHT_DECAY = 1e-5
EPOCHS = 30         
USE_AMP = True

# Multi-task loss weights (Maintain task dominance control)
LAMBDA_TUMOR  = 0.8  # Balanced for large ROI
LAMBDA_STROKE = 1.2  # Increased for sparse targets
LAMBDA_CLS    = 0.5  # Prevent Alzheimer dominance

# Joint Multi-Task Training Environment Ready

# Kaggle Roots
TUMOR_ROOT  = Path(os.environ.get("TUMOR_ROOT", "/kaggle/input/datasets/awsaf49/brats20-dataset-training-validation"))
STROKE_ROOT = Path(os.environ.get("STROKE_ROOT", "/kaggle/input/datasets/orvile/isles-2022-brain-stoke-dataset/ISLES-2022/ISLES-2022"))
ALZ_A_ROOT  = Path(os.environ.get("ALZ_A_ROOT", "/kaggle/input/datasets/summaiyamahmood/adni-preprocessed"))
ALZ_B_ROOT  = Path(os.environ.get("ALZ_B_ROOT", "/kaggle/input/datasets/summaiyamahmood/adni-677-sorted"))
TUMOR_ROOT_2021 = Path(os.environ.get("TUMOR_ROOT_2021", "/tmp/brats2021"))

CHECKPOINT_DIR = Path("./checkpoints_ablation")
CHECKPOINT_DIR.mkdir(exist_ok=True)
CACHE_VERSION = "v8_ablation_final"
CACHE_DIR = Path("/kaggle/working/cache_ablation")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEBUG = False

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ═══════════════════════════════════════════════════════════════════════════
#  2. SAFETY UTILS (Hard Fail System)
# ═══════════════════════════════════════════════════════════════════════════

def validate_tensor(x, name):
    """Hard Fail System (Section D): Ensure no NaNs or non-finite values."""
    if torch.isnan(x).any():
        raise RuntimeError(f"❌ HARD FAIL: {name} contains NaNs")
    if not torch.isfinite(x).all():
        raise RuntimeError(f"❌ HARD FAIL: {name} contains non-finite values")

def validate_dataset_stats(ds, name, is_seg=True):
    """Scientific Data Gate: Ensure meaningful signal and label presence."""
    if not ds: return False
    print(f"   🔍 Analyzing {name} ...")
    n_check = min(10, len(ds))
    indices = random.sample(range(len(ds)), n_check)
    all_means, all_stds = [], []
    mask_nonzero = 0
    for i in indices:
        try:
            b = ds[i]
            img = b["image"]
            all_means.append(img.mean().item())
            all_stds.append(img.std().item())
            if is_seg and b["seg"].sum() > 0:
                mask_nonzero += 1
        except Exception as e:
            print(f"      ⚠️  Sample {i} failed: {e}")
            continue
    if not all_means: return False
    avg_mean, avg_std = np.mean(all_means), np.mean(all_stds)
    print(f"      Stats: mean={avg_mean:.3f}, std={avg_std:.3f}")
    if abs(avg_mean) > 5.0 or avg_std < 0.1:
        raise RuntimeError(f"❌ HARD FAIL: {name} intensity collapse/shift detected (Check Preprocessing)")
    if is_seg:
        coverage = mask_nonzero / len(all_means)
        print(f"      Mask Coverage: {coverage:.1%} in checked sample")
        if coverage == 0.0 and len(ds) > 10:
            raise RuntimeError(f"❌ HARD FAIL: {name} masks are ALL empty (Check Dataset Loader)")
    return True

def load_nifti(path: Path) -> np.ndarray:
    try:
        img = nib.load(str(path))
        data = img.get_fdata()
        return np.asarray(data, dtype=np.float32)
    except Exception as e:
        raise RuntimeError(f"Failed to load NIfTI: {path} | Error: {e}")

def resize_volume(volume: torch.Tensor, is_mask: bool) -> torch.Tensor:
    """Helper: Resize to ROI_SIZE with appropriate interpolation."""
    if volume.ndim == 3:
        volume = volume.unsqueeze(0).unsqueeze(0)
    elif volume.ndim == 4:
        volume = volume.unsqueeze(0)
        
    resized = F.interpolate(
        volume,
        size=ROI_SIZE,
        mode="nearest" if is_mask else "trilinear",
        align_corners=None if is_mask else False
    )
    return resized.squeeze(0).float()

def universal_preprocess(volume: np.ndarray, is_mask: bool = False) -> torch.Tensor:
    """Unified robust preprocessing for Tumor and Stroke."""
    volume = volume.astype(np.float32)
    
    if not is_mask:
        p1, p99 = np.percentile(volume, (1, 99))
        volume = np.clip(volume, p1, p99)
        volume = (volume - p1) / (p99 - p1 + 1e-8)
        mean, std = volume.mean(), volume.std() + 1e-8
        volume = (volume - mean) / std
    
    vol_tensor = torch.from_numpy(volume).float()
    res_tensor = resize_volume(vol_tensor, is_mask).float()
    
    validate_tensor(res_tensor, "Preprocessed Volume")
    return res_tensor

def preprocess_alz_light(volume: np.ndarray) -> torch.Tensor:
    """Part 1: Light Alzheimer pipeline (separated from segmentation path)."""
    volume = volume.astype(np.float32)
    mean = volume.mean()
    std  = volume.std() + 1e-6
    volume = (volume - mean) / std
    volume = np.clip(volume, -3.0, 3.0)
    vol_t  = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)
    vol_t  = F.interpolate(vol_t, size=ROI_SIZE, mode="trilinear", align_corners=False)
    result = vol_t.squeeze(0)
    validate_tensor(result, "Alzheimer Light Preprocess")
    return result

# ═══════════════════════════════════════════════════════════════════════════
#  4. DATASETS
# ═══════════════════════════════════════════════════════════════════════════

class TumorDataset(Dataset):
    """BraTS 2020: T1ce Input -> [ET, NCR, ED] Independent Binary Masks"""
    def __init__(self, root_path: str, debug=False):
        self.root = Path(root_path)
        self.cases = self._find_cases()
        if debug:
            self.cases = self.cases[:20]
            print(f"⚠️ DEBUG: Tumor dataset restricted to {len(self.cases)} cases")
        print(f"✅ Tumor Dataset: {len(self.cases)} cases")

    def _find_cases(self):
        cases = []
        seen_dirs = set()
        for t1ce_file in self.root.rglob("*_t1ce.nii*"):
            if not t1ce_file.is_file() or t1ce_file.stat().st_size < 1024:
                continue
            parent = t1ce_file.parent
            if parent in seen_dirs:
                continue
            seen_dirs.add(parent)
            seg_candidates = [f for f in parent.glob("*_seg.nii*") if f.stat().st_size > 1024]
            flair_candidates = [f for f in parent.glob("*_flair.nii*") if f.stat().st_size > 1024]
            
            if seg_candidates and flair_candidates:
                cases.append({"t1ce": t1ce_file, "flair": flair_candidates[0], "seg": seg_candidates[0]})
            elif seg_candidates:
                cases.append({"t1ce": t1ce_file, "flair": None, "seg": seg_candidates[0]})
        return cases
    
    def __len__(self): return len(self.cases)
    
    def __getitem__(self, idx):
        case = self.cases[idx]
        cache_key = CACHE_DIR / f"{Path(case['t1ce']).stem}_{CACHE_VERSION}.pt"
        if cache_key.exists():
            try:
                data = torch.load(cache_key, weights_only=True)
                image, target = data["image"], data["target"]
                return {"image": image, "seg": target, "path": str(case['t1ce']), "presence": {"tumor": torch.tensor([1.0 if target.sum()>0 else 0.0])}, "has_label": {"tumor": torch.tensor([1.0])}}
            except Exception:
                cache_key.unlink(missing_ok=True)
        
        img_t1ce = universal_preprocess(load_nifti(case["t1ce"]), is_mask=False)
        if case.get("flair") and case["flair"] is not None and Path(case["flair"]).exists():
            img_flair = universal_preprocess(load_nifti(case["flair"]), is_mask=False)
            image = torch.cat([img_t1ce, img_flair], dim=0)
        else:
            image = torch.cat([img_t1ce, img_t1ce], dim=0)
            
        seg_vol = load_nifti(case["seg"])
        seg = universal_preprocess(seg_vol, is_mask=True).round()
        seg_et  = (seg == 4.0).float()   
        seg_ncr = (seg == 1.0).float()   
        seg_ed  = (seg == 2.0).float()   
        target = torch.cat([seg_et, seg_ncr, seg_ed], dim=0) 
        
        try:
            if shutil.disk_usage(str(CACHE_DIR)).free > 5 * 1024**3:
                torch.save({"image": image, "target": target}, cache_key)
        except Exception: pass
        
        return {"image": image, "seg": target, "path": str(case["t1ce"]), "presence": {"tumor": torch.tensor([1.0 if target.sum()>0 else 0.0])}, "has_label": {"tumor": torch.tensor([1.0])}}

class StrokeDataset(Dataset):
    """ISLES 2022: ADC Input -> Binary Mask Target (ADC only has masks)"""
    def __init__(self, root_path: str, debug=False):
        self.root = Path(root_path)
        self.cases = self._find_cases()
        if debug:
            self.cases = self.cases[:20]
            print(f"⚠️ DEBUG: Stroke dataset restricted to {len(self.cases)} cases")
        print(f"✅ Stroke Dataset: {len(self.cases)} cases")

    def _find_cases(self):
        cases = []
        all_stroke_dirs = list(self.root.rglob("sub-strokecase*"))
        subject_dirs = [d for d in all_stroke_dirs if d.is_dir()]
        
        for sub_dir in subject_dirs:
            adc_candidates = [f for f in sub_dir.rglob("*.nii*") if "adc" in f.name.lower() and f.is_file() and f.stat().st_size > 1024]
            if not adc_candidates: continue
            
            subj_name = sub_dir.name
            dataset_root = sub_dir.parent
            deriv_search = list((dataset_root / "derivatives" / subj_name).rglob("*msk*.nii*"))
            if not deriv_search:
                deriv_search = list(dataset_root.rglob(f"*{subj_name}*msk*.nii*"))
            if not deriv_search:
                deriv_search = list(adc_candidates[0].parent.rglob("*msk*.nii*"))
            
            valid_msk = [f for f in deriv_search if f.is_file() and f.stat().st_size > 1024]
            if not valid_msk: continue
            
            cases.append({"adc": adc_candidates[0], "msk": valid_msk[0]})
        return cases

    def __len__(self): return len(self.cases)

    def __getitem__(self, idx):
        case = self.cases[idx]
        cache_key = CACHE_DIR / f"{Path(case['adc']).stem}_{CACHE_VERSION}.pt"
        if cache_key.exists():
            try:
                data = torch.load(cache_key, weights_only=True)
                image, target = data["image"], data["target"]
                return {"image": image, "seg": target, "path": str(case["adc"]), "presence": {"stroke": torch.tensor([1.0 if target.sum()>0 else 0.0])}, "has_label": {"stroke": torch.tensor([1.0])}}
            except Exception:
                cache_key.unlink(missing_ok=True)
        
        vol_img = load_nifti(case["adc"])
        vol_msk = load_nifti(case["msk"])
        img_base = universal_preprocess(vol_img, is_mask=False)
        image = torch.cat([img_base, img_base], dim=0) 
        mask = universal_preprocess(vol_msk, is_mask=True)
        target = (mask > 0).float()
            
        try:
            if shutil.disk_usage(str(CACHE_DIR)).free > 5 * 1024**3:
                torch.save({"image": image, "target": target}, cache_key)
        except Exception: pass
        
        return {"image": image, "seg": target, "path": str(case["adc"]), "presence": {"stroke": torch.tensor([1.0 if target.sum()>0 else 0.0])}, "has_label": {"stroke": torch.tensor([1.0])}}

class AlzheimerDataset(Dataset):
    def __init__(self, records, dataset_type="preprocessed", augment=False):
        self.records      = records
        self.dataset_type = dataset_type
        self.augment      = augment
        self._cache_pfx   = f"alz_{dataset_type}_v8"

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        path_hash = _hashlib.md5(record["path"].encode()).hexdigest()[:8]
        cache_key = CACHE_DIR / f"{self._cache_pfx}_{path_hash}.npz"

        if cache_key.exists():
            try:
                data  = np.load(cache_key)
                image = torch.from_numpy(data["image"])
            except Exception:
                cache_key.unlink(missing_ok=True)
                return self.__getitem__(idx)
        else:
            vol = load_nifti(record["path"])
            image = preprocess_alz_light(vol)
            tmp = str(cache_key) + ".tmp.npz"
            np.savez_compressed(tmp, image=image.numpy())
            os.replace(tmp, cache_key)  

        if self.augment:
            if random.random() < 0.5: image = torch.flip(image, [2])
            if random.random() < 0.5: image = torch.flip(image, [3])
            if random.random() < 0.3: image = image + torch.randn_like(image) * 0.05
            if random.random() < 0.3: image = image * random.uniform(0.9, 1.1)

        return {"image": image, "seg": torch.zeros((1, *ROI_SIZE)), "presence": {"alzheimer": torch.tensor([float(record["label"])])}, "has_label": {"alzheimer": torch.tensor([1.0])}, "path": record["path"]}

def get_adni_subject_id(path: str) -> str:
    """Part A: Extract ADNI subject ID from path."""
    parts = path.replace("\\", "/").split("/")
    for p in parts:
        if "_S_" in p and len(p.split("_")) >= 3:
            return p
    fname = os.path.basename(path)
    return fname.split(".")[0]

def build_alzheimer_subject_split(alz_configs, seed=42, debug=False):
    """Part A: Unified subject-level train/val split across all ADNI roots."""
    from collections import defaultdict
    class_map = {"AD": 1, "CN": 0, "MCI": 0, "EMCI": 0, "LMCI": 0}
    all_samples = []

    for data_root_str, dataset_type in alz_configs:
        data_root = Path(data_root_str)
        if not data_root.exists(): continue

        print(f"  🔍 Scanning {dataset_type} Alzheimer dataset at {data_root.name}...")
        class_dirs_found = [c for c in data_root.rglob("*") if c.is_dir() and c.name in class_map]

        for cls_dir in class_dirs_found:
            label = class_map[cls_dir.name]
            nii_files = [f for f in cls_dir.rglob("*") if f.is_file() and (f.suffix == ".nii" or f.name.endswith(".nii.gz"))]
            for f in nii_files:
                all_samples.append({"path": str(f), "label": label, "type": dataset_type})

    if not all_samples:
        print("❌ CRITICAL: No Alzheimer samples found across any roots.")
        return [], []

    # De-duplicate
    seen_paths = set()
    deduped = []
    for s in all_samples:
        if s["path"] not in seen_paths:
            seen_paths.add(s["path"])
            deduped.append(s)
    all_samples = deduped

    if debug:
        random.shuffle(all_samples)
        all_samples = all_samples[:min(200, len(all_samples))]

    subject_map = defaultdict(list)
    for s in all_samples:
        sid = get_adni_subject_id(s["path"])
        subject_map[sid].append(s)

    subjects = list(subject_map.keys())
    subject_labels = []
    for sid in subjects:
        labels = [x["label"] for x in subject_map[sid]]
        subject_labels.append(max(set(labels), key=labels.count))

    try:
        train_subj, val_subj = train_test_split(subjects, test_size=0.2, stratify=subject_labels, random_state=seed)
    except Exception:
        train_subj, val_subj = train_test_split(subjects, test_size=0.2, random_state=seed)

    assert set(train_subj).isdisjoint(set(val_subj)), "CRITICAL: Subject leakage detected!"

    train_data, val_data = [], []
    for sid in train_subj: train_data.extend(subject_map[sid])
    for sid in val_subj: val_data.extend(subject_map[sid])

    print(f"✅ Alzheimer Split: {len(train_data)} train / {len(val_data)} val scans ({len(subjects)} subjects)")
    return train_data, val_data

# ═══════════════════════════════════════════════════════════════════════════
#  5. MODEL ARCHITECTURE (NeuroX-P Production Footprint)
# ═══════════════════════════════════════════════════════════════════════════

class SEBlock(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(c, c // r), nn.ReLU(inplace=True), nn.Linear(c // r, c), nn.Sigmoid())
    def forward(self, x):
        b, c = x.shape[0], x.shape[1]
        return x * self.fc(nn.AdaptiveAvgPool3d(1)(x).view(b, c)).view(b, c, 1, 1, 1)

class ResBlock3D(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv3d(ic, oc, 3, padding=1, bias=False), nn.InstanceNorm3d(oc, affine=True), nn.ReLU(inplace=True), nn.Conv3d(oc, oc, 3, padding=1, bias=False), nn.InstanceNorm3d(oc, affine=True))
        self.skip = nn.Conv3d(ic, oc, 1, bias=False) if ic != oc else nn.Identity()
    def forward(self, x): return F.relu(self.conv(x) + self.skip(x))

class TransformerBottleneck3D(nn.Module):
    def __init__(self, dim, depth, heads, mlp, drop=0.2):
        super().__init__()
        self.layers = nn.ModuleList([nn.ModuleList([nn.LayerNorm(dim), nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True), nn.LayerNorm(dim), nn.Sequential(nn.Linear(dim, mlp), nn.GELU(), nn.Dropout(drop), nn.Linear(mlp, dim), nn.Dropout(drop))]) for _ in range(depth)])
    def forward(self, x):
        b, c, d, h, w = x.shape
        if d*h*w > 2000: return x
        x = x.view(b, c, -1).permute(0, 2, 1)
        for ln1, attn, ln2, ff in self.layers:
            ao, _ = attn(ln1(x), ln1(x), ln1(x)); x = x + ao; x = x + ff(ln2(x))
        return x.permute(0, 2, 1).view(b, c, d, h, w)

class AttentionGate3D(nn.Module):
    def __init__(self, gc, sc, ic):
        super().__init__()
        self.Wg, self.Ws, self.psi = nn.Conv3d(gc, ic, 1), nn.Conv3d(sc, ic, 1), nn.Conv3d(ic, 1, 1)
    def forward(self, g, s): return s * torch.sigmoid(self.psi(F.relu(self.Wg(g) + self.Ws(s))))

class SegmentationDecoder(nn.Module):
    def __init__(self, out_c, name, use_att=True):
        super().__init__()
        self.name, self.use_att = name, use_att
        self.up3, self.up2, self.up1 = nn.ConvTranspose3d(128, 128, 2, 2), nn.ConvTranspose3d(128, 64, 2, 2), nn.ConvTranspose3d(64, 32, 2, 2)
        self.att3, self.att2, self.att1 = AttentionGate3D(128, 128, 64), AttentionGate3D(64, 64, 32), AttentionGate3D(32, 32, 16)
        self.dec3, self.dec2, self.dec1 = self._block(256, 128), self._block(128, 64), self._block(64, 32)
        self.head = nn.Conv3d(32, out_c, 1)
    def _block(self, ic, oc): return nn.Sequential(nn.Conv3d(ic, oc, 3, padding=1), nn.InstanceNorm3d(oc, affine=True), nn.ReLU(inplace=True), nn.Conv3d(oc, oc, 3, padding=1), nn.InstanceNorm3d(oc, affine=True), nn.ReLU(inplace=True))
    def forward(self, f, b):
        u3 = self.up3(b); d3 = self.dec3(torch.cat([u3, self.att3(u3, f["e3"]) if self.use_att else f["e3"]], 1))
        u2 = self.up2(d3); d2 = self.dec2(torch.cat([u2, self.att2(u2, f["e2"]) if self.use_att else f["e2"]], 1))
        u1 = self.up1(d2); d1 = self.dec1(torch.cat([u1, self.att1(u1, f["e1"]) if self.use_att else f["e1"]], 1))
        return self.head(d1)

class SharedEncoder(nn.Module):
    def __init__(self, inc=2):
        super().__init__()
        self.e1, self.e2, self.e3 = self._block(inc, 32), self._block(32, 64), self._block(64, 128)
        self.p1, self.p2, self.p3 = nn.MaxPool3d(2), nn.MaxPool3d(2), nn.MaxPool3d(2)
    def _block(self, ic, oc): return nn.Sequential(nn.Conv3d(ic, oc, 3, padding=1), nn.InstanceNorm3d(oc, affine=True), nn.ReLU(inplace=True), nn.Conv3d(oc, oc, 3, padding=1), nn.InstanceNorm3d(oc, affine=True), nn.ReLU(inplace=True))
    def forward(self, x):
        v1 = self.e1(x); v2 = self.e2(self.p1(v1)); v3 = self.e3(self.p2(v2))
        return {"e1": v1, "e2": v2, "e3": v3, "b": self.p3(v3)}

class PresenceHead(nn.Module):
    def __init__(self, inf=128):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(inf, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 2))
    def forward(self, b):
        o = self.fc(nn.AdaptiveAvgPool3d(1)(b).flatten(1))
        return o[:, 0:1], o[:, 1:2]

class AlzheimerEncoderV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1, self.b2, self.b3, self.b4 = ResBlock3D(1,32), ResBlock3D(32,64), ResBlock3D(64,128), ResBlock3D(128,256)
        self.p1, self.p2, self.p3 = nn.MaxPool3d(2), nn.MaxPool3d(2), nn.MaxPool3d(2)
        self.se = SEBlock(256) if ABLATION_CONFIG["USE_SE_BLOCK"] else nn.Identity()
        self.norm = nn.LayerNorm(512)
        self.cls = nn.Sequential(nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 1))
        self.lv = nn.Sequential(nn.Linear(512, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x):
        x = self.se(self.b4(self.p3(self.b3(self.p2(self.b2(self.p1(self.b1(x))))))))
        f = self.norm(torch.cat([nn.AdaptiveAvgPool3d(1)(x).flatten(1), nn.AdaptiveMaxPool3d(1)(x).flatten(1)], 1))
        return self.cls(f), self.lv(f)

class DecisionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, f): return self.mlp(f)

class NeuroXMultiDiseaseAblation(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder, self.bottleneck = SharedEncoder(2), TransformerBottleneck3D(128, 4, 8, 256) if ABLATION_CONFIG["USE_TRANSFORMER"] else nn.Identity()
        self.tumor_dec, self.stroke_dec = SegmentationDecoder(3, "tumor", ABLATION_CONFIG["USE_ATTENTION_GATES"]), SegmentationDecoder(1, "stroke", ABLATION_CONFIG["USE_ATTENTION_GATES"])
        self.tumor_pres, self.stroke_pres = PresenceHead(128), PresenceHead(128)
        self.alz_encoder = AlzheimerEncoderV3() if ABLATION_CONFIG["USE_ALZ_ISOLATION"] else nn.Linear(128, 1)
        self.decision_head, self.temp, self._epoch = DecisionHead(), nn.Parameter(torch.ones(1)), 0
    def forward(self, x, active_presence=None, active_seg=None):
        res = {"presence": {}, "segmentations": {}, "alz_logvar": None}
        if active_presence and "alzheimer" in active_presence:
            if ABLATION_CONFIG["USE_ALZ_ISOLATION"]:
                logits, lv = self.alz_encoder(x)
                res["presence"]["alzheimer"], res["alz_logvar"] = logits, lv
            else:
                f = self.encoder(torch.cat([x, x], 1))
                b = nn.AdaptiveAvgPool3d(1)(f["b"]).flatten(1)
                logits = self.alz_encoder(b)
                res["presence"]["alzheimer"] = logits
                res["alz_logvar"] = torch.zeros_like(logits)
        if (active_presence and any(k in ["tumor", "stroke"] for k in active_presence)) or (active_seg and any(k in ["tumor", "stroke"] for k in active_seg)):
            f = self.encoder(x); b_in = f["b"]
            if ABLATION_CONFIG["USE_TRANSFORMER"]: b = self.bottleneck(b_in)
            else: b = b_in
            if active_presence:
                if "tumor" in active_presence: res["presence"]["tumor"] = self.tumor_pres(b)
                if "stroke" in active_presence: res["presence"]["stroke"] = self.stroke_pres(b)
            if active_seg:
                if "tumor" in active_seg: res["segmentations"]["tumor"] = self.tumor_dec(f, b)
                if "stroke" in active_seg: res["segmentations"]["stroke"] = self.stroke_dec(f, b)
        if not self.training:
            t = self.temp.clamp(0.01, 10.0)
            for k in res["presence"]:
                v = res["presence"][k]
                if isinstance(v, tuple): res["presence"][k] = (v[0]/t, v[1])
                else: res["presence"][k] = v / t
        return res

# ═══════════════════════════════════════════════════════════════════════════
#  6. LOSS & EVALUATION (Scientific Parity)
# ═══════════════════════════════════════════════════════════════════════════

def compute_dice_loss(logits, targets):
    p = torch.sigmoid(logits); inter = (p * targets).sum((2,3,4)); union = p.sum((2,3,4)) + targets.sum((2,3,4))
    return 1.0 - ((2. * inter + 1.0) / (union + 1.0)).mean()

def compute_ece(probs, labels, n_bins=10):
    p, l = torch.as_tensor(probs), torch.as_tensor(labels).float()
    ece = 0.0; bins = torch.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        m = (p >= bins[i]) & (p < bins[i+1])
        if m.sum() > 0: ece += m.float().mean().item() * abs(l[m].mean().item() - p[m].mean().item())
    return ece

def dice_loss(pred, target, smooth=1e-6):
    """Differentiable Dice Loss for PyTorch training."""
    pred = pred.contiguous(); target = target.contiguous()
    intersection = (pred * target).sum(dim=(2, 3, 4))
    dice = (2. * intersection + smooth) / (pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4)) + smooth)
    return 1. - dice.mean()

def compute_dice_loss(p, g): return dice_loss(p, g)

def dice_score(p, g):
    ps, gs = p.sum(), g.sum()
    if ps == 0 and gs == 0: return 1.0
    return float((2. * (p * g).sum() + 1e-6) / (ps + gs + 1e-6))

def iou_score(p, g):
    inter = (p * g).sum(); union = p.sum() + g.sum() - inter
    if union == 0: return 1.0
    return float((inter + 1e-6) / (union + 1e-6))

def hausdorff95(p, g, s):
    p1, p2 = np.argwhere(p > 0).astype(float), np.argwhere(g > 0).astype(float)
    if not len(p1) and not len(p2): return 0.0
    if not len(p1) or not len(p2): return 100.0
    p1, p2 = p1 * np.array(s), p2 * np.array(s)
    t1, t2 = KDTree(p1), KDTree(p2)
    d1, _ = t2.query(p1); d2, _ = t1.query(p2)
    return float(max(np.percentile(d1, 95), np.percentile(d2, 95)))

def evaluate_segmentation_full(model, t_ld, s_ld, epoch):
    model.eval(); run_hd = (epoch % 5 == 0) or (epoch == EPOCHS)
    tm, sm = {"et_dice": [], "tc_dice": [], "wt_dice": [], "et_iou": [], "tc_iou": [], "wt_iou": [], "wt_hd95": []}, {"dice": [], "iou": [], "hd95": []}
    with torch.no_grad():
        for ld, m, tsk in [(t_ld, tm, "tumor"), (s_ld, sm, "stroke")]:
            if not ld: continue
            for b in ld:
                res = model(b["image"].to(DEVICE), active_seg=[tsk])
                p = torch.sigmoid(res["segmentations"][tsk]).cpu().numpy(); gt = b["seg"].cpu().numpy(); spc = (1.0, 1.0, 1.0)
                try: spc = nib.load(b["path"][0]).header.get_zooms()[:3]
                except: pass
                for i in range(p.shape[0]):
                    if tsk == "tumor":
                        p_et, gt_et = p[i, 0] > 0.5, gt[i, 0] > 0.5
                        p_tc, gt_tc = np.any(p[i, 0:2] > 0.5, 0), np.any(gt[i, 0:2] > 0.5, 0)
                        p_wt, gt_wt = np.any(p[i, 0:3] > 0.5, 0), np.any(gt[i, 0:3] > 0.5, 0)
                        
                        m["et_dice"].append(dice_score(p_et, gt_et)); m["tc_dice"].append(dice_score(p_tc, gt_tc)); m["wt_dice"].append(dice_score(p_wt, gt_wt))
                        m["et_iou"].append(iou_score(p_et, gt_et)); m["tc_iou"].append(iou_score(p_tc, gt_tc)); m["wt_iou"].append(iou_score(p_wt, gt_wt))
                        if run_hd: m["wt_hd95"].append(hausdorff95(p_wt, gt_wt, spc))
                    else:
                        ps, gs = p[i, 0] > 0.5, gt[i, 0] > 0.5
                        m["dice"].append(dice_score(ps, gs))
                        m["iou"].append(iou_score(ps, gs)) # Fix: Consistent smoothed IoU
                        if run_hd: m["hd95"].append(hausdorff95(ps, gs, spc))
    return {k: np.mean(v) if v else 0.0 for k, v in {**tm, **sm}.items()}

def evaluate_alzheimer_full(model, ld):
    model.eval(); y_t, y_p = [], []
    with torch.no_grad():
        for b in ld:
            o = model(b["image"].to(DEVICE), active_presence=["alzheimer"])
            y_t.append(b["presence"]["alzheimer"].numpy())
            lgt = o["presence"]["alzheimer"]
            if isinstance(lgt, tuple): lgt = lgt[0]
            y_p.append(torch.sigmoid(lgt).cpu().numpy())
    yt, yp = np.concatenate(y_t), np.concatenate(y_p)
    return {"auc": roc_auc_score(yt, yp), "auprc": average_precision_score(yt, yp), "brier": brier_score_loss(yt, yp), "ece": compute_ece(yp, yt)}

def train_step(model, b, tsk, opt, scaler):
    """Calculates loss for a single task without updating weights. Match parity with production loss functions."""
    img = b["image"].to(DEVICE)
    with torch.amp.autocast("cuda", enabled=USE_AMP):
        if tsk == "alzheimer":
            o = model(img, active_presence=["alzheimer"])
            lgt = o["presence"]["alzheimer"]
            tp = b["presence"]["alzheimer"].to(DEVICE).float().view(-1, 1)
            loss = F.binary_cross_entropy_with_logits(lgt, tp)
        else:
            o = model(img, active_seg=[tsk])
            sl = o["segmentations"][tsk]; gs = b["seg"].to(DEVICE)
            loss = 0.8 * compute_dice_loss(torch.sigmoid(sl), gs) + 0.2 * F.binary_cross_entropy_with_logits(sl, gs)
            
    if not torch.isfinite(loss): return None
    return loss

# ═══════════════════════════════════════════════════════════════════════════
#  7. MAIN LOOP (Mission Control)
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print(f"🚀 NEUROX ABLATION | SIMPLIFIED MODE | CONFIG: {ABLATION_CONFIG}")
    
    # 1. Unified Dataset Setup
    tra, vla = build_alzheimer_subject_split([(ALZ_A_ROOT, "A"), (ALZ_B_ROOT, "B")], SEED)
    d_alz_t = AlzheimerDataset(tra, "preprocessed", True)
    d_alz_v = AlzheimerDataset(vla, "preprocessed", False)
    dst, dss = TumorDataset(TUMOR_ROOT), StrokeDataset(STROKE_ROOT)
    
    print("\n📊 2. Hard Data Validation Gate")
    validate_dataset_stats(dst, "Tumor", True)
    validate_dataset_stats(dss, "Stroke", True)
    validate_dataset_stats(d_alz_v, "Alzheimer", False)
    
    # 2. DataLoaders
    loader_kwargs = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
    
    lda = DataLoader(d_alz_t, shuffle=True, **loader_kwargs)
    ldav = DataLoader(d_alz_v, shuffle=False, **loader_kwargs)
    tr_t, vl_t = train_test_split(list(range(len(dst))), test_size=0.2, random_state=SEED)
    tr_s, vl_s = train_test_split(list(range(len(dss))), test_size=0.2, random_state=SEED)
    ldt = DataLoader(torch.utils.data.Subset(dst, tr_t), shuffle=True, **loader_kwargs)
    ldtv = DataLoader(torch.utils.data.Subset(dst, vl_t), shuffle=False, **loader_kwargs)
    lds = DataLoader(torch.utils.data.Subset(dss, tr_s), shuffle=True, **loader_kwargs)
    ldsv = DataLoader(torch.utils.data.Subset(dss, vl_s), shuffle=False, **loader_kwargs)
    
    # 3. Model & Optim
    model = NeuroXMultiDiseaseAblation().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(enabled=USE_AMP)
    sched = CosineAnnealingLR(opt, T_max=50)
    
    history = []
    for ep in range(1, EPOCHS + 1):
        model._epoch = ep; model.train()
        it_a, it_t, it_s = iter(lda), iter(ldt), iter(lds)
        n_steps = max(len(lda), len(ldt), len(lds))
        
        pbar = tqdm(range(n_steps), desc=f"Ep{ep} [JOINT]")
        losses = []
        
        for i in pbar:
            opt.zero_grad()
            step_losses = []
            
            # Sub-Task 1: Alzheimer
            try: b = next(it_a)
            except StopIteration: it_a = iter(lda); b = next(it_a)
            l = train_step(model, b, "alzheimer", opt, scaler)
            if l is not None: step_losses.append(l)
            
            # Sub-Task 2: Tumor
            try: b = next(it_t)
            except StopIteration: it_t = iter(ldt); b = next(it_t)
            l = train_step(model, b, "tumor", opt, scaler)
            if l is not None: step_losses.append(l)
            
            # Sub-Task 3: Stroke
            try: b = next(it_s)
            except StopIteration: it_s = iter(lds); b = next(it_s)
            l = train_step(model, b, "stroke", opt, scaler)
            if l is not None: step_losses.append(l)
            
            if step_losses:
                total_loss = sum(step_losses) / len(step_losses)
                scaler.scale(total_loss).backward()
                scaler.step(opt); scaler.update()
                losses.append(total_loss.item())
                pbar.set_postfix(loss=f"{np.mean(losses[-10:]):.4f}", lr=f"{opt.param_groups[0]['lr']:.2e}")
        
        sched.step()
        
        # 🧪 Evaluation Phase
        model.eval()
        m_a = evaluate_alzheimer_full(model, ldav)
        m_s = evaluate_segmentation_full(model, ldtv, ldsv, ep)
        
        res = {**m_a, **m_s, "epoch": ep, "loss": np.mean(losses) if losses else 0.0}
        history.append(res); print(f"✅ Ep {ep} Metrics: {res}")
        
        pd.DataFrame(history).to_csv(CHECKPOINT_DIR / "ablation_metrics.csv", index=False)
        torch.save(model.state_dict(), CHECKPOINT_DIR / f"neurox_ablation_ep{ep}.pth")

if __name__ == "__main__": main()
