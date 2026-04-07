#%%writefile app.py

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
    precision_score, recall_score, f1_score, accuracy_score,
    roc_curve
)
from collections import Counter
import pandas as pd
from scipy.spatial.distance import cdist
import json # Fix 1: Top-level json import for safety
import shutil # Fix 1: Top-level shutil import for cache flushing

# ═══════════════════════════════════════════════════════════════════════════
#  0. GLOBAL SAFETY CHECKS & UTILS
# ═══════════════════════════════════════════════════════════════════════════
def validate_tensor(x, name):
    """Hard Fail System (Section D): Ensure no NaNs or non-finite values."""
    if torch.isnan(x).any():
        raise RuntimeError(f"❌ HARD FAIL: {name} contains NaNs")
    if not torch.isfinite(x).all():
        raise RuntimeError(f"❌ HARD FAIL: {name} contains non-finite values")

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
torch.backends.cudnn.benchmark = True  # Auto-tunes conv kernels for fixed (1,2,96,96,96) input
ROI_SIZE = (96, 96, 96)
BATCH_SIZE = 1
ACCUM_STEPS = 4
NUM_WORKERS = 2        
WEIGHT_DECAY = 1e-5
EPOCHS = 26         
USE_AMP = True       
USE_ATTENTION_GATES = False # Global Toggle (Ablation Finding: Gates hurt multi-task learning)
RESUME_PATH = "/kaggle/input/models/muntasaikarthik/last-checkpoint1/other/default/1/neurox_last.pth"   # Manual resume path (e.g., /kaggle/input/weights/neurox_last.pth)

# Section A: DETERMINISTIC ROOTS
TUMOR_ROOT  = Path(os.environ.get("TUMOR_ROOT", "/kaggle/input/datasets/awsaf49/brats20-dataset-training-validation"))
STROKE_ROOT = Path(os.environ.get("STROKE_ROOT", "/kaggle/input/datasets/orvile/isles-2022-brain-stoke-dataset/ISLES-2022/ISLES-2022"))
ATLAS_ROOT  = Path(os.environ.get("ATLAS_ROOT",  "/kaggle/input/datasets/muntasaikarthik/atlas-r2-dataset/ATLAS_2/Training"))
# Section A: DUAL ALZHEIMER ROOTS (Alz A: Preprocessed, Alz B: Raw/Sorted)
ALZ_A_ROOT  = Path(os.environ.get("ALZ_A_ROOT", "/kaggle/input/datasets/summaiyamahmood/adni-preprocessed"))
ALZ_B_ROOT  = Path(os.environ.get("ALZ_B_ROOT", "/kaggle/input/datasets/summaiyamahmood/adni-677-sorted"))
# Section A: Extra Tumor Roots (BraTS 2021 Optimization)
TUMOR_ROOT_2021 = Path(os.environ.get("TUMOR_ROOT_2021", "/tmp/brats2021"))

# Multi-task loss weights (Section B: task Dominance Control)
# L_total = 1.0 * L_alz + 0.7 * L_tumor + 0.7 * L_stroke
# Applied to prevent segmentation from dominating the shared encoder gradients.
LAMBDA_TUMOR  = 0.7
LAMBDA_STROKE = 0.7
LAMBDA_CLS    = 1.0

# Section B: PHASE-AWARE CURRICULUM
# Format: {upper_epoch_limit: (TRAIN_ALZ, TRAIN_SEG)}
PHASE_CONFIG = {
    10:  (True,  False),  # Phase 1: ALZ warmup (ep 1-6)
    26: (False, True),   # Phase 2: SEG warmup (ep 7-26, step-capped)
    48: (True,  True)    # Phase 3: Joint training (ep 27-31)
}

CHECKPOINT_DIR = Path("./checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Cache setup (Section C)
CACHE_VERSION   = "v7_2ch_flair"   # Tumor cache version (unchanged)
STROKE_CACHE_V  = "v8_modal"        # Stroke-specific: modality channel replaces duplicate
CACHE_DIR = Path("/kaggle/working/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
print(f"📦 Using persistent cache ({CACHE_VERSION}) at {CACHE_DIR}")

# DEBUG MODE
DEBUG = False  
if DEBUG:
    print("🔬 DEBUG MODE ENABLED: 3 epochs, restricted datasets.")
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
        raise RuntimeError(f"Failed to load NIfTI: {path} | Error: {e}")

def load_nifti_safe(path) -> np.ndarray:
    """Safe NIfTI loader: handles file paths AND directory paths (ATLAS-style structure)."""
    path = str(path)
    if os.path.isdir(path):
        for root_dir, _, files in os.walk(path):
            for f in files:
                if f.endswith(".nii") or f.endswith(".nii.gz"):
                    return nib.load(os.path.join(root_dir, f)).get_fdata().astype(np.float32)
        raise RuntimeError(f"No NIfTI file found in directory: {path}")
    return nib.load(path).get_fdata().astype(np.float32)

def build_atlas_cases(root_dir: Path) -> list:
    """Scan ATLAS R2 dataset and return valid T1w + lesion mask pairs.

    Filters cases with lesion mask < 50 voxels (too small for stable Dice gradient).
    Returns list of dicts: {'type': 'atlas', 'image': path, 'mask': path}
    """
    cases = []
    for root_str, _, _ in os.walk(str(root_dir)):
        if os.path.basename(root_str) == "anat":
            nii_files = []
            for r, _, files in os.walk(root_str):
                for f in files:
                    if f.endswith(".nii") or f.endswith(".nii.gz"):
                        nii_files.append(os.path.join(r, f))
            image, mask = None, None
            for f in nii_files:
                fl = f.lower()
                if "lesion" in fl:
                    mask = f
                elif "t1w" in fl or "norm" in fl:
                    image = f
            if image and mask:
                cases.append({"type": "atlas", "image": image, "mask": mask})
    filtered = []
    for c in cases:
        try:
            m = load_nifti_safe(c["mask"])
            if np.sum(m) > 50:
                filtered.append(c)
        except Exception:
            continue
    print(f"\u2705 ATLAS Builder: {len(cases)} found \u2192 {len(filtered)} after lesion filter (>50 voxels)")
    return filtered

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
    orig_sum = volume.sum()
    
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

def preprocess_alz(volume: np.ndarray, dataset_type: str) -> torch.Tensor:
    """Section B: Alzheimer Preprocessing (STRICT DETERMINISTIC PIPELINE)"""
    volume = volume.astype(np.float32)
    
    # Domain A: Unified Normalization (Fix 3)
    # Force z-score for network stability regardless of source dataset type
    mean, std = volume.mean(), volume.std() + 1e-8
    volume = (volume - mean) / std
    
    vol_tensor = torch.from_numpy(volume).float()
    res_tensor = resize_volume(vol_tensor, is_mask=False).float()
    validate_tensor(res_tensor, f"Alzheimer Volume ({dataset_type})")
    return res_tensor

def preprocess_alz_light(volume: np.ndarray) -> torch.Tensor:
    """Part 1: Light Alzheimer pipeline (separated from segmentation path).

    Uses pure z-score (no percentile clip) to preserve hippocampal contrast
    gradients, then soft-clips ±3σ to remove outliers without compressing
    mid-range signal. ROI size is identical (96³) — zero cache format break.
    """
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
        """Deep recursive finder — works regardless of nesting depth or wrapper folders.
        
        Strategy: rglob for ALL *_t1ce.nii* files anywhere under root, then locate
        the matching *_seg.nii* in the same directory. This handles:
          - BraTS20_Training_XXX/file.nii.gz  (standard)
          - wrapper/BraTS20_Training_XXX/file.nii.gz  (Kaggle extra nesting)
          - Datasets where folder name ≠ BraTS pattern but files follow naming convention
        """
        cases = []
        seen_dirs = set()
        # rglob finds t1ce files at ANY depth
        for t1ce_file in self.root.rglob("*_t1ce.nii*"):
            if not t1ce_file.is_file() or t1ce_file.stat().st_size < 1024:
                continue
            parent = t1ce_file.parent
            if parent in seen_dirs:
                continue
            seen_dirs.add(parent)
            seg_candidates = [f for f in parent.glob("*_seg.nii*") if f.stat().st_size > 1024]
            # 🧩 Domain B: FLAIR Matching
            flair_candidates = [f for f in parent.glob("*_flair.nii*") if f.stat().st_size > 1024]
            
            if seg_candidates and flair_candidates:
                cases.append({"t1ce": t1ce_file, "flair": flair_candidates[0], "seg": seg_candidates[0]})
            elif seg_candidates:
                cases.append({"t1ce": t1ce_file, "flair": None, "seg": seg_candidates[0]})
        
        if not cases:
            print(f"  ⚠️  TumorDataset: No t1ce+seg pairs found (size > 1KB) under {self.root}")
        return cases
    
    def __len__(self): return len(self.cases)
    
    def __getitem__(self, idx):
        case = self.cases[idx]
        # Section C: SAFE CACHE (v7 torch-native)
        cache_key = CACHE_DIR / f"{Path(case['t1ce']).stem}_{CACHE_VERSION}.pt"
        if cache_key.exists():
            try:
                data = torch.load(cache_key, weights_only=True)
                image, target = data["image"], data["target"]
                # Verify shape handles 1 or 2 channels (Domain B)
                if image.shape[1:] != ROI_SIZE or image.shape[0] not in (1, 2) or target.shape != (3, *ROI_SIZE) or torch.isnan(image).any():
                    raise ValueError("Shape mismatch")
                return {"image": image, "seg": target, "presence": {"tumor": torch.tensor([1.0])}, "has_label": {"tumor": torch.tensor([1.0])}, "path": str(case['t1ce'])}
            except Exception:
                cache_key.unlink(missing_ok=True)
        
        # Loading Modalities (T1ce + FLAIR)
        img_t1ce = universal_preprocess(load_nifti(case["t1ce"]), is_mask=False)
        if case.get("flair") and case["flair"] is not None and Path(case["flair"]).exists():
            img_flair = universal_preprocess(load_nifti(case["flair"]), is_mask=False)
            image = torch.cat([img_t1ce, img_flair], dim=0)   # shape (2, 96, 96, 96)
        else:
            image = torch.cat([img_t1ce, img_t1ce], dim=0)    # fallback: duplicate channel
            
        seg_vol = load_nifti(case["seg"])
        seg = universal_preprocess(seg_vol, is_mask=True).round()
        
        # BraTS Labels: 1=NCR, 2=ED, 4=ET
        seg_et  = (seg == 4.0).float()   
        seg_ncr = (seg == 1.0).float()   
        seg_ed  = (seg == 2.0).float()   
        target = torch.cat([seg_et, seg_ncr, seg_ed], dim=0) 
        
        # Save to cache with disk-safe guard (Change 16)
        try:
            if shutil.disk_usage(str(CACHE_DIR)).free > 5 * 1024**3:
                torch.save({"image": image, "target": target}, cache_key)
        except Exception:
            pass
        
        has_tumor = 1.0 if target.sum() > 0 else 0.0
        
        return {
            "image": image,
            "seg": target,
            "has_seg": torch.tensor([1.0]),
            "path": str(case["t1ce"]), # Fix: Added path for spacing extraction
            "presence": {
                "tumor": torch.tensor([has_tumor]),
                "stroke": torch.tensor([0.0]),
                "alzheimer": torch.tensor([0.0])
            },
            "has_label": {
                "tumor": torch.tensor([1.0]),
                "stroke": torch.tensor([0.0]),
                "alzheimer": torch.tensor([0.0])
            }
        }

# ═══════════════════════════════════════════════════════════════════════════
#  3. DATASET ADAPTERS (SECTION 3 REFACTORED)
# ═══════════════════════════════════════════════════════════════════════════

class StrokeDataset(Dataset):
    """ISLES 2022: DWI/ADC Input -> Binary Mask Target"""
    def __init__(self, root_path: str, extra_cases: list = None, debug=False):
        self.root = Path(root_path)
        self.cases = self._find_cases()
        if extra_cases:
            self.cases = self.cases + list(extra_cases)
            print(f"  ➕ Merged {len(extra_cases)} ATLAS cases. Total stroke: {len(self.cases)}")
        if debug:
            self.cases = self.cases[:30]
            print(f"⚠️ DEBUG: Stroke dataset restricted to {len(self.cases)} cases")
        print(f"✅ Stroke Dataset: {len(self.cases)} cases (ISLES + ATLAS)")

    def _find_cases(self):
        """Flexible deep finder for ISLES-2022 structure at any nesting depth.
        
        Probes for sub-strokecase* directories recursively so it works whether
        the dataset is mounted as:
          root/ISLES-2022/ISLES-2022/sub-strokecase.../  (standard)
          root/sub-strokecase.../                         (flat)
          root/extra_wrapper/ISLES-2022/...               (Kaggle variant)
        """
        cases = []
        
        # Find all subject directories named sub-strokecase* anywhere under root
        all_stroke_dirs = list(self.root.rglob("sub-strokecase*"))
        subject_dirs = [d for d in all_stroke_dirs if d.is_dir()]
        
        if not subject_dirs:
            print(f"  ⚠️  StrokeDataset: No sub-strokecase* dirs found under {self.root}")
            all_nii = list(self.root.rglob("*.nii*"))[:5]
            for f in all_nii:
                print(f"      Found: {f}")
            return cases

        for sub_dir in subject_dirs:
            # Find ADC image: search recursively within this subject directory
            adc_candidates = [
                f for f in sub_dir.rglob("*.nii*")
                if "adc" in f.name.lower() and f.is_file() and f.stat().st_size > 1024
            ]
            if not adc_candidates:
                continue
            
            # Find mask: look for msk file — could be in derivatives sibling or within subject dir
            # Strategy 1: derivatives/<subject>/ses-*/msk
            subj_name = sub_dir.name
            # Walk up to find the dataset root (parent of sub-strokecase siblings)
            dataset_root = sub_dir.parent
            # Try standard derivatives path
            deriv_search = list((dataset_root / "derivatives" / subj_name).rglob("*msk*.nii*"))
            if not deriv_search:
                # Try looking anywhere in dataset root derivatives
                deriv_search = list(dataset_root.rglob(f"*{subj_name}*msk*.nii*"))
            if not deriv_search:
                # Last resort: any msk file co-located with the adc file
                deriv_search = list(adc_candidates[0].parent.rglob("*msk*.nii*"))
            
            valid_msk = [f for f in deriv_search if f.is_file() and f.stat().st_size > 1024]
            if not valid_msk:
                continue
            
            cases.append({"adc": adc_candidates[0], "msk": valid_msk[0]})
        
        return cases

    def __len__(self): return len(self.cases)

    def __getitem__(self, idx):
        case = self.cases[idx]
        is_atlas = case.get("type") == "atlas"

        # Cache key — use the correct image path stem for ISLES vs ATLAS
        img_path_str = str(case["image"]) if is_atlas else str(case["adc"])
        cache_key = CACHE_DIR / f"{Path(img_path_str).stem}_{STROKE_CACHE_V}.pt"

        if cache_key.exists():
            try:
                data = torch.load(cache_key, weights_only=True)
                image, target = data["image"], data["target"]
                if image.shape[1:] != ROI_SIZE or image.shape[0] not in (1, 2) or torch.isnan(image).any():
                    raise ValueError("Shape mismatch — expected 2-ch modality image")
                has_stroke = 1.0 if target.sum() > 0 else 0.0
                return {"image": image, "seg": target, "has_seg": torch.tensor([1.0]),
                        "path": img_path_str,
                        "presence": {"tumor": torch.tensor([0.0]), "stroke": torch.tensor([has_stroke]), "alzheimer": torch.tensor([0.0])},
                        "has_label": {"tumor": torch.tensor([0.0]), "stroke": torch.tensor([1.0]), "alzheimer": torch.tensor([0.0])}}
            except Exception:
                cache_key.unlink(missing_ok=True)

        if is_atlas:
            # ATLAS: T1w image + lesion mask. Z-score ONLY (prevents ATLAS/ISLES identity learning)
            vol_img = load_nifti_safe(case["image"]).astype(np.float32)
            vol_msk = load_nifti_safe(case["mask"]).astype(np.float32)
            vol_img = (vol_img - np.mean(vol_img)) / (np.std(vol_img) + 1e-8)
            vol_img = np.clip(vol_img, -5, 5)  # Clamp: removes extreme artifacts, stabilizes cross-dataset training
            img_tensor = torch.from_numpy(vol_img).float()
            img_base = resize_volume(img_tensor, is_mask=False)
            validate_tensor(img_base, "ATLAS Stroke Volume")
        else:
            # ISLES: ADC image — standard percentile clip + z-score pipeline
            vol_img = load_nifti(case["adc"])
            vol_msk = load_nifti(case["msk"])
            img_base = universal_preprocess(vol_img, is_mask=False)

        # Modality channel: ATLAS=1.0 (T1w source), ISLES=0.0 (ADC source)
        # Replaces the meaningless duplicate second channel — encoder now knows the input domain
        modality_ch = torch.ones_like(img_base) if is_atlas else torch.zeros_like(img_base)
        image = torch.cat([img_base, modality_ch], dim=0)  # (2, 96, 96, 96)
        mask = universal_preprocess(vol_msk, is_mask=True)
        target = (mask > 0).float()

        # Save to cache with disk-safe guard
        try:
            if shutil.disk_usage(str(CACHE_DIR)).free > 5 * 1024**3:
                torch.save({"image": image, "target": target}, cache_key)
        except Exception:
            pass

        has_stroke = 1.0 if target.sum() > 0 else 0.0
        return {
            "image": image,
            "seg": target,
            "has_seg": torch.tensor([1.0]),
            "path": img_path_str,
            "presence": {
                "tumor": torch.tensor([0.0]),
                "stroke": torch.tensor([has_stroke]),
                "alzheimer": torch.tensor([0.0])
            },
            "has_label": {
                "tumor": torch.tensor([0.0]),
                "stroke": torch.tensor([1.0]),
                "alzheimer": torch.tensor([0.0])
            }
        }

import hashlib as _hashlib

class AlzheimerDataset(Dataset):
    """Alzheimer dataset with caching and per-dataset normalization.

    dataset_type='preprocessed' (Dataset A: ADNI preprocessed)
        Pipeline: z-score via preprocess_volume → resize to ROI_SIZE.
    dataset_type='sorted' (Dataset B: ADNI sorted / raw scanner)
        Pipeline: percentile clip [p1,p99] → [0,1] → [-1,1] → resize ONLY.
        preprocess_volume z-score is intentionally SKIPPED to avoid
        double-normalization cancelling the robust scaling (ISSUE 2).

    Cache keys are MD5(file_path)[:8] — stable across dataset reordering (ISSUE 3).
    """
    def __init__(self, records, dataset_type="preprocessed", augment=False):
        self.records      = records
        self.dataset_type = dataset_type
        self.augment      = augment
        # Part 5: Local cache version — invalidates ONLY alz .npz files.
        # Tumor/stroke .pt caches use CACHE_VERSION (v7_2ch_flair) and are untouched.
        ALZ_CACHE_V  = "v8_alz_light"
        self._cache_pfx = f"alz_A_{ALZ_CACHE_V}" if dataset_type == "preprocessed" else f"alz_B_{ALZ_CACHE_V}"

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]

        # ISSUE 3 FIX: path-hash cache key — stable even if dataset order changes
        path_hash = _hashlib.md5(record["path"].encode()).hexdigest()[:8]
        cache_key = CACHE_DIR / f"{self._cache_pfx}_{path_hash}.npz"

        # Section C: SAFE CACHE (v4)
        if cache_key.exists():
            try:
                data  = np.load(cache_key)
                image = torch.from_numpy(data["image"])
                if image.shape != (1, *ROI_SIZE) or torch.isnan(image).any():
                    raise ValueError("Corrupt cache")
            except Exception:
                cache_key.unlink(missing_ok=True)
                return self.__getitem__(idx)
        else:
            vol = load_nifti(record["path"])
            # Part 2: Light pipeline separation — replaces shared preprocessing
            image = preprocess_alz_light(vol)

            # Change 533-535: Use unique temporary filename to avoid cache race conditions (ISSUE 3 FIX)
            import uuid
            _tmp = f"{str(cache_key)}_{uuid.uuid4().hex}.tmp.npz"
            try:
                np.savez_compressed(_tmp, image=image.numpy())
                os.replace(_tmp, cache_key)
            except Exception:
                if os.path.exists(_tmp):
                    os.remove(_tmp)

        if self.augment:
            if random.random() < 0.5:
                image = torch.flip(image, [2])
            if random.random() < 0.5:
                image = torch.flip(image, [3])
            # G1: Gaussian noise + intensity scaling (post-cache, never pollutes cache)
            if random.random() < 0.3:
                image = image + torch.randn_like(image) * 0.05
            if random.random() < 0.3:
                image = image * random.uniform(0.9, 1.1)

        return {
            "image": image,
            "seg": torch.zeros((1, *ROI_SIZE)),
            "has_seg": torch.tensor([0.0]),
            "presence": {
                "tumor":     torch.tensor([0.0]),
                "stroke":    torch.tensor([0.0]),
                "alzheimer": torch.tensor([float(record["label"])])
            },
            "has_label": {
                "tumor":     torch.tensor([0.0]),
                "stroke":    torch.tensor([0.0]),
                "alzheimer": torch.tensor([1.0])
            }
        }

def get_adni_subject_id(path: str) -> str:
    """Part A: Extract ADNI subject ID from path.
    
    Handles:
    - Deep preprocessed structure: .../002_S_0413/... -> '002_S_0413'
    - Flat files: 002_S_0619.nii -> '002_S_0619'
    """
    parts = path.replace("\\", "/").split("/")
    # Case 1: look for folder named with ADNI pattern (XXX_S_XXXX)
    for p in parts:
        if "_S_" in p and len(p.split("_")) >= 3:
            return p
    # Case 2: flat file — use stem before first dot
    fname = os.path.basename(path)
    return fname.split(".")[0]

def build_alzheimer_subject_split(alz_configs, seed=42, debug=False):
    """Part A: Unified subject-level train/val split across all ADNI roots.
    
    Merges ALL sources BEFORE splitting to prevent subject leakage.
    Subject ID extracted via get_adni_subject_id() to handle both
    preprocessed (deep) and sorted (flat) dataset structures.
    """
    from collections import defaultdict
    class_map = {"AD": 1, "CN": 0, "MCI": 0, "EMCI": 0, "LMCI": 0}
    all_samples = []

    for data_root_str, dataset_type in alz_configs:
        data_root = Path(data_root_str)
        if not data_root.exists():
            print(f"  ⚠️  Alzheimer root skipped (not found): {data_root}")
            continue

        print(f"  🔍 Scanning {dataset_type} Alzheimer dataset at {data_root.name}...")
        class_dirs_found = [
            c for c in data_root.rglob("*")
            if c.is_dir() and c.name in class_map
        ]

        if not class_dirs_found:
            print(f"  ❌ No class directories (AD/CN/MCI/EMCI/LMCI) found under {data_root}")
            continue

        for cls_dir in class_dirs_found:
            label = class_map[cls_dir.name]
            nii_files = [
                f for f in cls_dir.rglob("*")
                if f.is_file() and (f.suffix == ".nii" or f.name.endswith(".nii.gz"))
            ]
            for f in nii_files:
                all_samples.append({
                    "path":  str(f),
                    "label": label,
                    "type":  dataset_type,
                })

    if not all_samples:
        print("\u274c CRITICAL: No Alzheimer samples found across any roots.")
        return [], []

    # De-duplicate by path (handles overlapping mounts)
    seen_paths = set()
    deduped = []
    for s in all_samples:
        if s["path"] not in seen_paths:
            seen_paths.add(s["path"])
            deduped.append(s)
    all_samples = deduped

    if debug:
        import random as _rnd
        _rnd.shuffle(all_samples)
        all_samples = all_samples[:min(200, len(all_samples))]

    # A3: Group by subject (merge BEFORE split to prevent leakage)
    subject_map = defaultdict(list)
    for s in all_samples:
        sid = get_adni_subject_id(s["path"])
        subject_map[sid].append(s)

    subjects = list(subject_map.keys())

    # A4: Stratify split on majority label per subject
    subject_labels = []
    for sid in subjects:
        labels = [x["label"] for x in subject_map[sid]]
        subject_labels.append(max(set(labels), key=labels.count))

    try:
        train_subj, val_subj = train_test_split(
            subjects, test_size=0.2, stratify=subject_labels, random_state=seed
        )
    except Exception:
        train_subj, val_subj = train_test_split(subjects, test_size=0.2, random_state=seed)

    # A6: MANDATORY leakage check
    assert set(train_subj).isdisjoint(set(val_subj)), \
        "CRITICAL: Subject leakage detected between train/val splits!"

    # A5: Build flat sample lists
    train_data, val_data = [], []
    for sid in train_subj:
        train_data.extend(subject_map[sid])
    for sid in val_subj:
        val_data.extend(subject_map[sid])

    ad_count = sum(1 for s in all_samples if s["label"] == 1)
    cn_count  = len(all_samples) - ad_count
    print(f"✅ Alzheimer Multi-Source Split Summary:")
    print(f"   {len(train_subj)} train subjects / {len(val_subj)} val subjects")
    print(f"   AD={ad_count} CN={cn_count} total scans: {len(all_samples)}")
    print(f"   train={len(train_data)} val={len(val_data)}")

    # A: Rename 'path' key to match what AlzheimerDataset expects (uses record['path'])
    # Already using 'path' — safe.
    return train_data, val_data


# ═══════════════════════════════════════════════════════════════════════════
#  4. MODEL ARCHITECTUREand
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
        # Mandatory Runtime Check
        
             
        e1, e2, e3 = enc_features["enc1"], enc_features["enc2"], enc_features["enc3"]
        b = bottleneck_features # This is the output of the TransformerBottleneck3D
        u3 = self.up3(b)
        
        # A: Bypass Attention Gates (Global Toggle)
        if USE_ATTENTION_GATES:
            d3 = self.dec3(torch.cat([u3, self.att3(u3, e3)], dim=1))
        else:
            d3 = self.dec3(torch.cat([u3, e3], dim=1))
        
        u2 = self.up2(d3)
        if USE_ATTENTION_GATES:
            d2 = self.dec2(torch.cat([u2, self.att2(u2, e2)], dim=1))
        else:
            d2 = self.dec2(torch.cat([u2, e2], dim=1))
            
        u1 = self.up1(d2)
        if USE_ATTENTION_GATES:
            d1 = self.dec1(torch.cat([u1, self.att1(u1, e1)], dim=1))
        else:
            d1 = self.dec1(torch.cat([u1, e1], dim=1))
        
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

        # Optimization: Only run alz_encoder when presence["alzheimer"] is requested
        if active_presence and "alzheimer" in active_presence:
            alz_logits, alz_log_var = self.alz_encoder(x)
            res["presence"]["alzheimer"] = alz_logits
            res["alzheimer_log_var"] = alz_log_var
            
        # Shared Path (Seg + Presence) - Only run if Tumor/Stroke requested
        if (active_presence and any(k in ["tumor", "stroke"] for k in active_presence)) or \
           (active_seg and any(k in ["tumor", "stroke"] for k in active_seg)):
            
            feats = self.encoder(x)
            # Fix #9: Smooth transformer alpha-blend (prevents hard-switch gradient shock)
            # alpha linearly ramps 0→1 over epochs 1-6, then full transformer from epoch 7
            _epoch = getattr(self, '_current_epoch', 99)
            alpha = min(1.0, _epoch / 6.0)
            if alpha < 1.0:
                raw = feats["bottleneck_input"].detach()  # gradient-free bypass component
                bottleneck_feats = alpha * self.bottleneck(feats["bottleneck_input"]) + (1.0 - alpha) * raw
            else:
                bottleneck_feats = self.bottleneck(feats["bottleneck_input"])
            
            # Presence Heads (Calibrated via global temperature)
            temp = self.temperature.clamp(0.01, 10.0)
            
            if active_presence and "tumor" in active_presence:
                 res["presence"]["tumor"] = self.tumor_presence(bottleneck_feats)
            if active_presence and "stroke" in active_presence:
                 res["presence"]["stroke"] = self.stroke_presence(bottleneck_feats)
                 
            # Segmentation Decoders
            if active_seg and "tumor" in active_seg:
                res["segmentations"]["tumor"] = self.tumor_decoder(feats, bottleneck_feats)
            if active_seg and "stroke" in active_seg:
                res["segmentations"]["stroke"] = self.stroke_decoder(feats, bottleneck_feats)

        # Part D: Apply Temperature Scaling ONLY during eval (not training)
        # During training, logit gradients should flow unscaled through the loss
        if not self.training:
            for k in res["presence"]:
                if isinstance(res["presence"][k], tuple):
                    logit, log_var = res["presence"][k]
                    res["presence"][k] = (logit / self.temperature.clamp(0.01, 10.0), log_var)
                else:
                    res["presence"][k] = res["presence"][k] / self.temperature.clamp(0.01, 10.0)

        return res

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

def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """Focal Loss: down-weights easy voxels, forces focus on hard lesion boundaries.
    Fixes stroke Dice plateau by generating gradient for hard negative/positive voxels.
    alpha=0.25, gamma=2.0 are standard ISLES/BraTS tuned values.
    """
    probs = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    pt = probs * targets + (1 - probs) * (1 - targets)  # p_t: correct class probability
    loss = alpha * (1 - pt) ** gamma * bce
    return loss.mean()

def quick_dice(probs, targets):
    """Soft Dice Score (0-1) for logging. Expects sigmoid probabilities [0,1], NOT logits."""
    # Do NOT apply sigmoid here — callers pass torch.sigmoid(output) directly.
    inter = (probs * targets).sum(dim=(2, 3, 4))
    union = probs.sum(dim=(2, 3, 4)) + targets.sum(dim=(2, 3, 4))
    dice = (2. * inter) / (union + 1e-6)
    return dice.mean().item()


def compute_ece(probs, labels, n_bins=10):
    """Expected Calibration Error: Strictly bin-based reliability measure.
    
    🧩 Domain C: Scientific Alzheimer Metrics (ECE)
    """
    if not isinstance(probs, torch.Tensor):
        probs = torch.from_numpy(probs)
    if not isinstance(labels, torch.Tensor):
        labels = torch.from_numpy(labels).float()
        
    bins = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    ece = torch.zeros(1, device=probs.device)

    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i+1])
        if mask.sum() > 0:
            acc = labels[mask].float().mean()
            conf = probs[mask].mean()
            # Weight the bin error by its frequency in the dataset
            ece += (mask.float().mean()) * torch.abs(acc - conf)

    return float(ece.item())

# ═══════════════════════════════════════════════════════════════════════════
#  RESEARCH-GRADE SEGMENTATION METRICS
# ═══════════════════════════════════════════════════════════════════════════

def dice_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    """Dice Similarity Coefficient. Handles empty masks."""
    pred_sum = pred.sum()
    gt_sum   = gt.sum()
    if pred_sum == 0 and gt_sum == 0:
        return 1.0
    if pred_sum == 0 or gt_sum == 0:
        return 0.0
    intersection = (pred * gt).sum()
    return float((2.0 * intersection + eps) / (pred_sum + gt_sum + eps))

def iou_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    """Intersection over Union. Handles empty masks."""
    pred_sum = pred.sum()
    gt_sum   = gt.sum()
    if pred_sum == 0 and gt_sum == 0:
        return 1.0
    if pred_sum == 0 or gt_sum == 0:
        return 0.0
    intersection = (pred * gt).sum()
    union = pred_sum + gt_sum - intersection
    return float((intersection + eps) / (union + eps))

def hausdorff95(pred: np.ndarray, gt: np.ndarray,
                spacing: tuple = (1.0, 1.0, 1.0),
                fallback: float = 100.0) -> float:
    """95th-percentile Hausdorff Distance in PHYSICAL SPACE (mm).
    
    spacing: voxel size in mm along each axis (D, H, W).
    Without spacing the metric is in voxel units and becomes
    dataset-dependent — scientifically invalid for reporting.
    """
    pred_pts = np.argwhere(pred > 0).astype(np.float64)
    gt_pts   = np.argwhere(gt   > 0).astype(np.float64)
    if len(pred_pts) == 0 and len(gt_pts) == 0:
        return 0.0
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return fallback
    # H1 FIX: Hard-stop if too many points for cdist (memory/time explosion)
    if len(pred_pts) > 5000 or len(gt_pts) > 5000:
        return fallback
    # Convert voxel indices to mm coordinates
    spacing_arr = np.array(spacing, dtype=np.float64)
    pred_pts = pred_pts * spacing_arr
    gt_pts   = gt_pts   * spacing_arr
    distances = cdist(pred_pts, gt_pts, metric='euclidean')
    hd95 = max(
        np.percentile(np.min(distances, axis=1), 95),
        np.percentile(np.min(distances, axis=0), 95)
    )
    return float(hd95)

def average_surface_distance(pred: np.ndarray, gt: np.ndarray,
                             spacing: tuple = (1.0, 1.0, 1.0),
                             fallback: float = 100.0) -> float:
    """Average Symmetric Surface Distance in PHYSICAL SPACE (mm)."""
    pred_pts = np.argwhere(pred > 0).astype(np.float64)
    gt_pts   = np.argwhere(gt   > 0).astype(np.float64)
    if len(pred_pts) == 0 and len(gt_pts) == 0:
        return 0.0
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return fallback
    # Fix 6: Guard ASD against large lesioned GT masks (OOM safety)
    if len(pred_pts) > 5000 or len(gt_pts) > 5000:
        return fallback
    spacing_arr = np.array(spacing, dtype=np.float64)
    pred_pts = pred_pts * spacing_arr
    gt_pts   = gt_pts   * spacing_arr
    distances = cdist(pred_pts, gt_pts, metric='euclidean')
    asd = (np.mean(np.min(distances, axis=1)) + np.mean(np.min(distances, axis=0))) / 2.0
    return float(asd)

def volume_error(pred: np.ndarray, gt: np.ndarray) -> float:
    """Relative Volume Error |Vpred - Vgt| / Vgt. Returns nan if gt is empty."""
    gt_vol = float(gt.sum())
    if gt_vol == 0:
        return float('nan')
    return float(abs(float(pred.sum()) - gt_vol) / gt_vol)

# How often to run the expensive HD95/ASD pass (in epochs)
EVAL_HD_EVERY = 5

def evaluate_segmentation_full(model, tumor_val_loader, stroke_val_loader, epoch: int) -> dict:
    """Full segmentation validation: Dice, IoU, HD95 (mm), ASD (mm), VolumeError.
    
    🧩 Domain B: Research-Grade Metrics (TC, WT) and Temperature Calibration.
    """
    model.eval()
    run_hd = (epoch % EVAL_HD_EVERY == 0)

    # Accumulators
    tumor_metrics  = {"et_dice": [], "tc_dice": [], "wt_dice": [],
                      "tc_iou":  [], "wt_iou":  [], "tc_ve":   [], "wt_ve":   [],
                      "et_hd95": [], "wt_hd95": [], "et_asd": [], "wt_asd": []}
    stroke_metrics = {"dice": [], "iou": [], "ve": [], "hd95": [], "asd": []}

    with torch.no_grad():
        # ── Tumor VAL ──────────────────────────────────────────────────────────────
        if tumor_val_loader:
            for batch in tumor_val_loader:
                img = batch["image"].to(DEVICE)
                gt_all = batch["seg"].cpu().numpy()          # [B, 3, D, H, w]
                
                # Domain G: Dynamic Spacing
                path = batch.get("path", [None])[0]
                spacing = (1.0, 1.0, 1.0)
                if path and Path(path).exists():
                    try: spacing = nib.load(str(path)).header.get_zooms()[:3]
                    except Exception: pass
                
                out = model(img, active_presence=["tumor"], active_seg=["tumor"])
                # Fix 3: Standardize segmentation to threshold 0.5 on raw sigmoids (Maintain benchmark comparability)
                probs = torch.sigmoid(out["segmentations"]["tumor"]).cpu().numpy()
                
                for b in range(img.shape[0]):
                    gt_b   = (gt_all[b] > 0.5).astype(np.uint8)  # [3, D, H, W]

                    # 🧩 Domain B: Tumor Core (TC) and Whole Tumor (WT)
                    # ET: Enhancing Tumor (Ch 0)
                    # TC: Tumor Core (NCR + ET) (Ch 1 + Ch 0)
                    # WT: Whole Tumor (NCR + ET + ED) (Ch 1 + Ch 0 + Ch 2)
                    p_et = (probs[b, 0] > 0.5).astype(np.uint8)
                    p_tc = (np.any(probs[b, 0:2] > 0.5, axis=0)).astype(np.uint8)
                    p_wt = (np.any(probs[b, 0:3] > 0.5, axis=0)).astype(np.uint8)
                    
                    gt_et = gt_b[0]
                    gt_tc = (np.any(gt_b[0:2] > 0.5, axis=0)).astype(np.uint8)
                    gt_wt = (np.any(gt_b[0:3] > 0.5, axis=0)).astype(np.uint8)
                    
                    tumor_metrics["et_dice"].append(dice_score(p_et, gt_et))
                    tumor_metrics["tc_dice"].append(dice_score(p_tc, gt_tc))
                    tumor_metrics["wt_dice"].append(dice_score(p_wt, gt_wt))
                    
                    tumor_metrics["tc_iou"].append(iou_score(p_tc, gt_tc))
                    tumor_metrics["wt_iou"].append(iou_score(p_wt, gt_wt))
                    tumor_metrics["tc_ve"].append(volume_error(p_tc, gt_tc))
                    tumor_metrics["wt_ve"].append(volume_error(p_wt, gt_wt))
                    
                    if run_hd:
                        tumor_metrics["et_hd95"].append(hausdorff95(p_et, gt_et, spacing=spacing))
                        tumor_metrics["wt_hd95"].append(hausdorff95(p_wt, gt_wt, spacing=spacing))
                        tumor_metrics["et_asd"].append(average_surface_distance(p_et, gt_et, spacing=spacing))
                        tumor_metrics["wt_asd"].append(average_surface_distance(p_wt, gt_wt, spacing=spacing))

        # ── Stroke VAL ──────────────────────────────────────────────────────────────
        if stroke_val_loader:
            for batch in stroke_val_loader:
                img = batch["image"].to(DEVICE)
                gt_np = batch["seg"].cpu().numpy()               # [B, 1, D, H, W]
                
                out = model(img, active_presence=["stroke"], active_seg=["stroke"])
                # Fix 3: Standardize segmentation to threshold 0.5 on raw sigmoids
                probs = torch.sigmoid(out["segmentations"]["stroke"]).cpu().numpy()

                # Domain G: Dynamic Spacing
                path = batch.get("path", [None])[0]
                spacing = (1.0, 1.0, 1.0)
                if path and Path(path).exists():
                    try: spacing = nib.load(str(path)).header.get_zooms()[:3]
                    except Exception: pass

                for b in range(img.shape[0]):
                    g = (gt_np[b, 0] > 0.5).astype(np.uint8)
                    p_cal = (probs[b, 0] > 0.5).astype(np.uint8)

                    # Step 8: Foreground-aware — skip background-only samples (prevents 0.50 plateau bias)
                    if g.sum() > 0:
                        stroke_metrics["dice"].append(dice_score(p_cal, g))
                        stroke_metrics["iou"].append(iou_score(p_cal, g))
                        stroke_metrics["ve"].append(volume_error(p_cal, g))

                        if run_hd:
                            stroke_metrics["hd95"].append(hausdorff95(p_cal, g, spacing=spacing))
                            stroke_metrics["asd"].append(average_surface_distance(p_cal, g, spacing=spacing))

    model.train()
    
    _agg = lambda x: np.nanmean(x) if len(x) > 0 else 0.0
    t = tumor_metrics
    s = stroke_metrics
    agg = {
        #Overlap (Scientific Research Set)
        "tumor_et_dice":  _agg(t["et_dice"]),
        "tumor_tc_dice":  _agg(t["tc_dice"]),
        "tumor_wt_dice":  _agg(t["wt_dice"]),
        "tumor_tc_iou":   _agg(t["tc_iou"]),
        "tumor_wt_iou":   _agg(t["wt_iou"]),
        "tumor_tc_ve":    _agg(t["tc_ve"]),
        "tumor_wt_ve":    _agg(t["wt_ve"]),
        
        "stroke_dice":    _agg(s["dice"]),
        "stroke_iou":     _agg(s["iou"]),
        "stroke_ve":      _agg(s["ve"]),
        
        # Boundary (Physical mm)
        "tumor_et_hd95":  _agg(t["et_hd95"]), "tumor_wt_hd95":  _agg(t["wt_hd95"]),
        "stroke_hd95":    _agg(s["hd95"]),
        "tumor_et_asd":   _agg(t["et_asd"]),  "tumor_wt_asd":   _agg(t["wt_asd"]),
        "stroke_asd":     _agg(s["asd"]),
    }

    # Pretty print
    print(f"\n  📐 SEG VALIDATION (Scientific Metric Set | Calibrated)")
    print(f"  Tumor  | ET={agg['tumor_et_dice']:.4f} TC={agg['tumor_tc_dice']:.4f} WT={agg['tumor_wt_dice']:.4f}")
    print(f"  Stroke | Dice={agg['stroke_dice']:.4f} IoU={agg['stroke_iou']:.4f}")
    
    # Hall of fame logic helper
    agg["tumor_mean"] = (agg["tumor_et_dice"] + agg["tumor_tc_dice"] + agg["tumor_wt_dice"]) / 3
    
    return agg

def evaluate_alzheimer_full(model, val_loader) -> dict:
    """Full Alzheimer validation: AUROC, AUPRC, Brier Score, ECE.
    
    🧩 Domain C: Scientific Alzheimer Metrics (AUROC, AUPRC, Brier Score)
    """
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for batch in val_loader:
            img = batch["image"].to(DEVICE)
            target = batch["presence"]["alzheimer"].to(DEVICE)
            mask = batch["has_label"]["alzheimer"].to(DEVICE)
            
            # Fix 2: Alzheimer validation uses standard forward path for consistency
            out = model(img, active_presence=["alzheimer"])
            logit = out["presence"]["alzheimer"]
            # Temperature scaling is applied inside forward() when not training
            prob = torch.sigmoid(logit)
            
            y_true.append(target[mask.bool()].cpu().numpy())
            y_prob.append(prob[mask.bool()].cpu().numpy())
            
    model.train()
    
    # Domain B: Scientific Alzheimer Metrics (AUROC, AUPRC, Brier Score)
    # Binary metrics (F1, Accuracy, Precision, Recall) use 0.5 threshold
    try:
        y_true = np.concatenate(y_true)
        y_prob = np.concatenate(y_prob)
        y_pred = (y_prob > 0.5).astype(float)
        
        # 1. Probabilistic Benchmarks
        auc_score = roc_auc_score(y_true, y_prob)
        auprc     = average_precision_score(y_true, y_prob)
        brier     = brier_score_loss(y_true, y_prob)
        ece       = compute_ece(y_prob, y_true)
        
        # 2. Categorical Benchmarks (Threshold-based)
        from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
        f1       = f1_score(y_true, y_pred)
        acc      = accuracy_score(y_true, y_pred)
        prec     = precision_score(y_true, y_pred, zero_division=0)
        rec      = recall_score(y_true, y_pred, zero_division=0)
        
        return {
            "auc":       auc_score,
            "auprc":     auprc,
            "brier":     brier,
            "ece":       ece,
            "f1":        f1,
            "accuracy":  acc,
            "precision": prec,
            "recall":    rec
        }
    except Exception as e:
        print(f"⚠️ Validation error (pathology empty?): {e}")
        return {
            "auc": 0.5, "auprc": 0.5, "brier": 0.25, "ece": 0.0,
            "f1": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0
        }

def train_step(model, batch, task, optimizer, scaler, epoch,
               use_uncertainty=True, use_decision=True, ep_decision_weight=1.0):
    img = batch["image"].to(DEVICE)
    
    # E3: AMP for seg tasks only (fp16 exp() overflows on ALZ)
    amp_enabled = USE_AMP and (task in ["tumor", "stroke"])

    # Fix #5: Gradient isolation — freeze shared encoder during ALZ backward
    # Prevents ALZ classification gradients from corrupting seg encoder weights
    if task == "alzheimer":
        for p in model.encoder.parameters():
            p.requires_grad = False

    with torch.amp.autocast("cuda", enabled=amp_enabled):
        if task == "alzheimer":
            out = model(img, active_presence=["alzheimer"])
            calibrated_logit = out["presence"]["alzheimer"]
            # Part 2: Logical shape enforcement — prevents silent broadcasting
            calibrated_logit = torch.clamp(calibrated_logit.view(-1, 1), -5, 5)  # FIX 3: Tighter clamp
            log_var = torch.clamp(out["alzheimer_log_var"], -0.5, 0.5)
            
            # Part 2: Logical shape enforcement — prevents silent broadcasting
            target_pres = batch["presence"]["alzheimer"].to(DEVICE).float().view(-1, 1)
            # FIX 2: Label smoothing (Diversity Hardening)
            target_pres = target_pres * 0.9 + 0.05
            
            mask = batch["has_label"]["alzheimer"].to(DEVICE)
            
            # Part 1: No pos_weight — Sampler handles imbalance; loss stays neutral to avoid distortion
            loss_pres = F.binary_cross_entropy_with_logits(
                calibrated_logit, target_pres, reduction='none')
            loss_pres = (loss_pres * mask).sum() / (mask.sum() + 1e-6)
            # Fix #8: explicit L2 removed — AdamW weight_decay=1e-4 handles regularization
            total_loss = LAMBDA_CLS * loss_pres
            
        elif task in ["tumor", "stroke"]:
            res = model(img, active_presence=[task], active_seg=[task])
            logit, log_var = res["presence"][task]
            log_var = torch.clamp(log_var, -0.5, 0.5)
            seg_logits = res["segmentations"][task]
            
            target_pres = batch["presence"][task].to(DEVICE).float()
            tgt = batch["seg"].to(DEVICE)
            mask = batch["has_label"][task].to(DEVICE)
            
            bce = F.binary_cross_entropy_with_logits(logit, target_pres, reduction='none')
            # D3: Staged uncertainty
            if use_uncertainty:
                loss_pres = bce * torch.exp(-log_var.detach()) + 0.05 * log_var
            else:
                loss_pres = bce
            loss_pres = (loss_pres * mask).sum() / (mask.sum() + 1e-6)
            
            loss_dice = compute_dice_loss(seg_logits, tgt).mean()
            if task == "stroke":
                loss_bce = F.binary_cross_entropy_with_logits(
                    seg_logits, tgt, pos_weight=STROKE_POS_WEIGHT).mean()
            else:
                loss_bce = F.binary_cross_entropy_with_logits(seg_logits, tgt).mean()
            loss_focal = focal_loss(seg_logits, tgt)
            loss_seg = loss_dice + 0.5 * loss_bce + 0.3 * loss_focal  # Dice+BCE+Focal (hard voxel fix)
            
            lam = LAMBDA_TUMOR if task == "tumor" else LAMBDA_STROKE
            total_loss = lam * loss_seg + LAMBDA_CLS * loss_pres
            
            # D3: Decision head — only active in joint phase
            if use_decision:
                with torch.no_grad():
                    seg_p = torch.sigmoid(seg_logits / model.temperature.clamp(0.01, 10.0))
                    feat_prob = torch.sigmoid(logit).view(-1, 1)
                    feat_unc  = torch.exp(log_var).view(-1, 1)
                    feat_vol  = seg_p.sum(dim=(1,2,3,4)).view(-1, 1) / (96**3)
                    if seg_logits.shape[1] > 1:
                        ps = torch.softmax(seg_logits.detach(), dim=1)
                        ent_map = -(ps * torch.log(ps + 1e-8)).sum(dim=1, keepdim=True)
                    else:
                        ps = seg_p.detach().clamp(1e-8, 1 - 1e-8)
                        ent_map = -(ps * torch.log(ps) + (1 - ps) * torch.log(1 - ps))
                    feat_entropy = ent_map.mean(dim=(1,2,3,4), keepdim=False).view(-1, 1)
                    feat_conf = seg_p.amax(dim=(2,3,4)).mean(dim=1, keepdim=True)
                    inter = (seg_p * tgt).sum(dim=(1,2,3,4))
                    union = seg_p.sum(dim=(1,2,3,4)) + tgt.sum(dim=(1,2,3,4))
                    soft_dice = torch.where(union < 1e-5, torch.zeros_like(inter), (2.*inter)/(union+1e-6))
                    correct_target = soft_dice.unsqueeze(1)
                
                features = torch.cat([feat_prob.detach(), feat_unc.detach(),
                                      feat_vol, feat_entropy, feat_conf], dim=1)
                dec_pred = model.decision_head(features)
                loss_dec = F.binary_cross_entropy_with_logits(dec_pred, correct_target)
                # Fix #10: decision_weight applied into total_loss
                total_loss = total_loss + ep_decision_weight * loss_dec

    # Fix #5: Restore shared encoder gradients after ALZ forward
    if task == "alzheimer":
        for p in model.encoder.parameters():
            p.requires_grad = True

    # Part B: Fix Loss Gate (Task-specific thresholds)
    if task == "alzheimer":
        threshold = 4.0
    elif task == "stroke":
        threshold = 12.0
    elif task == "tumor":
        threshold = 8.0
    else:
        threshold = 8.0

    if not torch.isfinite(total_loss) or total_loss.item() > threshold:
        optimizer.zero_grad(set_to_none=True)
        return None
    
    return total_loss


def check_batches(tumor_loader, stroke_loader):
    """Pre-training batch sanity check. Skips gracefully on empty loaders."""
    print("\n🔍 PRE-TRAINING BATCH CHECK 🔍")
    if tumor_loader is None or len(tumor_loader) == 0:
        print("⚠️ Tumor loader is empty — skipping batch check.")
    else:
        try:
            batch = next(iter(tumor_loader))
            img, seg = batch["image"], batch["seg"]
            print(f"✅ Tumor Batch: Img {img.shape} Range [{img.min():.2f}, {img.max():.2f}] | Seg {seg.shape} Sum {seg.sum().item():.0f}")
            if seg.sum() < 1:
                print("⚠️ WARNING: Tumor mask sum is 0!")
        except Exception as e:
            print(f"❌ Tumor Batch Check Failed: {e}")

    if stroke_loader is None or len(stroke_loader) == 0:
        print("⚠️ Stroke loader is empty — skipping batch check.")
    else:
        try:
            batch = next(iter(stroke_loader))
            img, seg = batch["image"], batch["seg"]
            print(f"✅ Stroke Batch: Img {img.shape} Range [{img.min():.2f}, {img.max():.2f}] | Seg {seg.shape} Sum {seg.sum().item():.0f}")
            if seg.sum() < 1:
                print("⚠️ WARNING: Stroke mask sum is 0 (Small lesion lost in resize?)")
        except Exception as e:
            print(f"❌ Stroke Batch Check Failed: {e}")
    print("="*40 + "\n")

# ═══════════════════════════════════════════════════════════════════════════
#  6. MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════

def main():
    global USE_AMP, STROKE_POS_WEIGHT

    # R3 FIX: Persistence over destruction (Fix 1)
    print("📦 Using persistent cache for rapid session resume...")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # STEP 6: DATA DISCOVERY (Section 1)
    # Pre-initialize loaders to None (Domain A: Runtime Fix)
    tumor_loader, stroke_loader, alz_loader = None, None, None
    tumor_val_loader, stroke_val_loader, alz_val_loader = None, None, None
    tumor_train_ds, stroke_train_ds, alz_train_ds = None, None, None
    tumor_val_ds, stroke_val_ds, alz_val_ds = None, None, None
    
    # Audit Fix 1: Initialize scheduler to None to prevent potential NameError
    scheduler_alz = None
    
    print(f"\n🔍 1. Discovery: Paths are task-isolated ...")
    
    # Tumor (Dual Root: 2020 + 2021 Optimization)
    try:
        tumor_datasets = []
        for troot in [TUMOR_ROOT, TUMOR_ROOT_2021]:
            if troot.exists():
                try:
                    ds = TumorDataset(troot, debug=DEBUG)
                    if len(ds) > 0:
                        tumor_datasets.append(ds)
                        print(f"   ✅ Loaded tumor from {troot.name}: {len(ds)} cases")
                except Exception as e:
                    print(f"   ⚠️ Tumor root skipped: {troot.name} | {e}")
        
        if tumor_datasets:
            from torch.utils.data import ConcatDataset
            tumor_ds_full = ConcatDataset(tumor_datasets) if len(tumor_datasets) > 1 else tumor_datasets[0]
            print(f"✅ Combined Tumor Dataset: {len(tumor_ds_full)} total cases")
            idx_t = range(len(tumor_ds_full))
            train_idx, val_idx = train_test_split(list(idx_t), test_size=0.2, random_state=SEED)
            tumor_train_ds = torch.utils.data.Subset(tumor_ds_full, train_idx)
            tumor_val_ds   = torch.utils.data.Subset(tumor_ds_full, val_idx)
        else:
            tumor_train_ds, tumor_val_ds = None, None
            print("⚠️ No tumor data found.")
    except Exception as e: print(f"⚠️ Serious Tumor Load Error: {e}")

    # Stroke (ISLES 2022 + ATLAS R2 merged)
    try:
        print(f"   Building ATLAS cases from {ATLAS_ROOT}...")
        atlas_cases = build_atlas_cases(ATLAS_ROOT) if ATLAS_ROOT.exists() else []
        stroke_ds_full = StrokeDataset(STROKE_ROOT, extra_cases=atlas_cases, debug=DEBUG)
        if len(stroke_ds_full) > 0:
            idx_s = list(range(len(stroke_ds_full)))
            train_idx_s, val_idx_s = train_test_split(idx_s, test_size=0.2, random_state=SEED)
            stroke_train_ds = torch.utils.data.Subset(stroke_ds_full, train_idx_s)
            stroke_val_ds   = torch.utils.data.Subset(stroke_ds_full, val_idx_s)
            # Step 7: Weighted sampler -- ISLES=1.0, ATLAS=0.3 (ISLES=lesion definition, ATLAS=diversity)
            from torch.utils.data import WeightedRandomSampler as _WRS
            _sw = [0.3 if stroke_ds_full.cases[i].get("type") == "atlas" else 1.0 for i in train_idx_s]
            stroke_loader = DataLoader(stroke_train_ds, batch_size=BATCH_SIZE,
                                       sampler=_WRS(_sw, len(_sw), replacement=True),
                                       num_workers=NUM_WORKERS, pin_memory=True,
                                       persistent_workers=True)
    except Exception as e: print(f"⚠️ Stroke Load Error: {e}")

    # Alzheimer (Hybrid Load: Alz A + Alz B)
    try:
        # Load both preprocessed (A) and raw/sorted (B) cohorts
        alz_configs = [(ALZ_A_ROOT, "preprocessed"), (ALZ_B_ROOT, "raw")]
        alz_train_rec, alz_val_rec = build_alzheimer_subject_split(alz_configs, seed=SEED, debug=DEBUG)
        
        if alz_train_rec:
            # AlzheimerDataset now uses the 'type' field from each record for per-sample normalization
            alz_train_ds = AlzheimerDataset(alz_train_rec, augment=True)
            alz_val_ds   = AlzheimerDataset(alz_val_rec,   augment=False)
            print(f"   🧬 Alzheimer hybrid cohort built with {len(alz_train_ds)} train samples.")
    except Exception as e:
        print(f"⚠️ Alzheimer Load Error: {e}")

    # Domain H: Hard stop if NO datasets loaded
    if not any([tumor_train_ds, stroke_train_ds, alz_train_ds]):
        raise RuntimeError("❌ HARD STOP: No valid datasets found across any task.")

    # STEP 8: CREATE DATALOADERS
    loader_kwargs = dict(batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    val_kwargs    = dict(batch_size=2, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

    tumor_loader  = DataLoader(tumor_train_ds, **loader_kwargs) if tumor_train_ds else None
    # stroke_loader already created above with WeightedRandomSampler (ISLES=1.0, ATLAS=0.5)
    if stroke_loader is None and stroke_train_ds is not None:
        stroke_loader = DataLoader(stroke_train_ds, **loader_kwargs)
    
    # Part 3: WeightedRandomSampler — ensures balanced AD/CN across all batches
    if alz_train_ds:
        from torch.utils.data import WeightedRandomSampler
        alz_labels       = [r["label"] for r in alz_train_ds.records]
        class_counts     = np.bincount(alz_labels)
        class_weights    = 1.0 / (class_counts + 1e-6)
        sample_weights   = [class_weights[l] for l in alz_labels]
        # Phase-aware oversampling:
        # Fresh start (no checkpoint): 2x = 1932 steps — encoder needs max gradient signal during ALZ warmup
        # Resumed run (checkpoint exists): 1x = 966 steps — warmup done, keeps max_steps ≤ 1000 for joint phase
        _alz_oversample = 1 if (RESUME_PATH and Path(RESUME_PATH).exists()) else 2
        alz_sampler      = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights) * _alz_oversample,
            replacement=True
        )
        alz_loader = DataLoader(
            alz_train_ds, batch_size=BATCH_SIZE,
            sampler=alz_sampler,
            num_workers=NUM_WORKERS, pin_memory=True,
            persistent_workers=True
        )
    else:
        alz_loader = None
    
    tumor_val_loader  = DataLoader(tumor_val_ds, **val_kwargs) if tumor_val_ds else None
    stroke_val_loader = DataLoader(stroke_val_ds, **val_kwargs) if stroke_val_ds else None
    alz_val_loader     = DataLoader(alz_val_ds,   **val_kwargs) if alz_val_ds else None

    # hard validation and sanity logs (Fix 4 / 6)
    print("\n📊 2. Hard Data Validation Gate")
    
    def validate_dataset_stats(ds, name, is_seg=True):
        if not ds: return False
        print(f"   🔍 Analyzing {name} ...")
        n_check = min(10, len(ds))
        indices = random.sample(range(len(ds)), n_check)
        
        all_means, all_stds = [], []
        mask_nonzero_counts = 0
        
        for i in indices:
            batch = ds[i]
            img = batch["image"]
            all_means.append(img.mean().item())
            all_stds.append(img.std().item())
            if is_seg:
                if batch["seg"].sum() > 0:
                    mask_nonzero_counts += 1
        
        avg_mean = np.mean(all_means)
        avg_std  = np.mean(all_stds)
        print(f"      Stats: mean={avg_mean:.3f}, std={avg_std:.3f}")
        
        # Hard Stop: Intensity Collapse
        if abs(avg_mean) > 5.0 or avg_std < 0.1:
            raise RuntimeError(f"❌ HARD STOP: {name} intensity stats pathological (Collapse/Shift detected)")
        
        # Hard Stop: Mask Missing (if seg dataset)
        if is_seg:
            coverage = mask_nonzero_counts / n_check
            print(f"      Mask Coverage: {coverage:.1%} in checked sample")
            if coverage == 0.0 and len(ds) > 10:
                raise RuntimeError(f"❌ HARD STOP: {name} masks are ALL empty in preview. Check discovery/preprocessing.")
        return True

    validate_dataset_stats(tumor_train_ds, "Tumor", is_seg=True)
    validate_dataset_stats(stroke_train_ds, "Stroke", is_seg=True)
    validate_dataset_stats(alz_train_ds, "Alzheimer", is_seg=False)

    # Alzheimer Balance check
    if alz_train_ds:
        all_alz_labels = [r["label"] for r in alz_train_ds.records]
        alz_pos = int(sum(all_alz_labels))
        alz_neg = len(all_alz_labels) - alz_pos
        pos, neg = alz_pos, alz_neg
        print(f"   ⚖️ ALZ Balance: AD={alz_pos}, CN={alz_neg}")
    else:
        alz_pos, alz_neg = 236, 972  # fallback constants
        pos, neg = alz_pos, alz_neg

    if stroke_train_ds:
        STROKE_POS_WEIGHT = torch.tensor([5.0], device=DEVICE)  # Reduced from 20.0 (was causing BCE instability)
    else:
        STROKE_POS_WEIGHT = torch.tensor([1.0], device=DEVICE)

    TUMOR_SPACING  = (1.0, 1.0, 1.0)   
    STROKE_SPACING = (1.0, 1.0, 1.0)

    # ═══════════════════════════════════════════════════════════════════════════
    #  7. OPTIMIZER & SCALER (PHASE-AWARE GROUPS)
    # ═══════════════════════════════════════════════════════════════════════════

    # 3. Setup Model & Optimizers (Enforced 2-Channel for FLAIR+T1ce)
    model = NeuroXMultiDisease(in_channels=2).to(DEVICE)

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

    # Fix 1: Phase-aware LR — epoch 1 is always Phase 1, so set baseline lr
    # lr_alz will be recomputed each epoch at the top of the epoch loop
    optimizer_alz = torch.optim.AdamW(model.alz_encoder.parameters(), lr=3e-5, weight_decay=1e-4)
    # scheduler_alz removed — conflicts with explicit phase-aware LR control
    scheduler_shared = None

    # Initial scalers
    # NOTE: scaler_alz is disabled — Alzheimer train_step runs in fp32 (no autocast).
    # Tumor/stroke use AMP for memory efficiency on T4.
    scaler_shared = torch.amp.GradScaler(enabled=USE_AMP)
    scaler_alz    = torch.amp.GradScaler(enabled=False)  # Alzheimer always fp32

    scheduler_shared = None

    print(f"🚀 Starting Training (DEBUG={DEBUG}, EPOCHS={EPOCHS})...")

    # ─── DATASET SANITY CHECKS (Section 9) ──────────────────────────────────
    def dataset_sanity_check(ds, name):
        if not ds or len(ds) == 0: return
        sample = ds[0]
        img = sample["image"]
        if torch.isnan(img).any():
            print(f"📊 {name} Sanity Check: ❌ CRITICAL: NaN detected in first sample")
        else:
            print(f"📊 {name} Sanity: shape={img.shape}, range=[{img.min():.2f}, {img.max():.2f}]")

    dataset_sanity_check(tumor_train_ds, "Tumor")
    dataset_sanity_check(stroke_train_ds, "Stroke")
    dataset_sanity_check(alz_train_ds, "Alzheimer")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Metrics History
    # ─────────────────────────────────────────────────────────────────────────
    metrics_history = {
        "epoch": [],
        "train": {
            "tumor": {"loss": [], "et_dice": [], "mean_dice": []},
            "stroke": {"loss": [], "dice": []},
            "alz":   {"loss": []},
        },
        "val": {
            "tumor": {
                "et_dice":  [], "tc_dice":  [], "wt_dice":  [],
                "tc_iou":   [], "wt_iou":   [],
                "et_hd95":  [], "wt_hd95":  [],
                "et_asd":   [], "wt_asd":   [],
                "tc_ve":    [], "wt_ve":    [],
            },
            "stroke": {
                "dice":  [], "iou":  [],
                "hd95":  [], "asd":  [],
                "ve":    [],
            },
            "alz": {
                "auc":       [], "auprc":     [], "brier":     [], "ece":      [],
                "f1":        [], "accuracy":  [], "precision": [], "recall":   [],
            },
        },
        "meta": {
            "score":      [],
            "best_epoch": None,
            "best_score": None,
        },
    }
    
    # Change 20: Robust Checkpoint Resume System
    # Priority: 1. Manual RESUME_PATH -> 2. Persistent 'neurox_last.pth' -> 3. Fallback 'neurox_model.pth'
    start_epoch = 1
    checkpoint_to_load = None
    if RESUME_PATH and Path(RESUME_PATH).exists():
        print(f"🔄 Resuming from {RESUME_PATH}...")
        # Fix 5: Ensure checkpoint_to_load is set for RESUME_PATH branch
        checkpoint_to_load = Path(RESUME_PATH)
    elif (CHECKPOINT_DIR / "neurox_last.pth").exists():
        checkpoint_to_load = CHECKPOINT_DIR / "neurox_last.pth"
    elif (CHECKPOINT_DIR / "neurox_model.pth").exists():
        checkpoint_to_load = CHECKPOINT_DIR / "neurox_model.pth"
        
    if checkpoint_to_load:
        print(f"\n🔁 Found existing checkpoint: {checkpoint_to_load}")
        try:
            ckpt = torch.load(checkpoint_to_load, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer_shared.load_state_dict(ckpt["optimizer_shared"])
            optimizer_alz.load_state_dict(ckpt["optimizer_alz"])
            if "scaler_shared" in ckpt: scaler_shared.load_state_dict(ckpt["scaler_shared"])
            if "scaler_alz" in ckpt: scaler_alz.load_state_dict(ckpt["scaler_alz"])
            
            # Robust metrics resume: merge loaded metrics with current structure to avoid KeyErrors
            loaded_metrics = ckpt.get("metrics", {})
            for top_k in ["train", "val"]:
                if top_k in loaded_metrics:
                    for task_k in loaded_metrics[top_k]:
                        if task_k in metrics_history[top_k]:
                            for metric_k in loaded_metrics[top_k][task_k]:
                                if metric_k in metrics_history[top_k][task_k]:
                                    metrics_history[top_k][task_k][metric_k] = loaded_metrics[top_k][task_k][metric_k]
            
            start_epoch = ckpt["epoch"] + 1
            print(f"✅ Successfully resumed from epoch {ckpt['epoch']}. Next: {start_epoch}")
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint: {e}. Starting fresh.")

    recovery_triggered = False  # Fix 4: guard against repeated adaptive boosts
    for epoch in range(start_epoch, EPOCHS + 1):
        global LAMBDA_CLS
        USE_AMP = True

        # Fix 1: Phase-aware LR for ALZ optimizer
        # ep27-29: 3-epoch re-warm burst after optimizer reset to escape collapse basin fast
        # ep30+:   conservative joint learning rate
        if epoch <= 10:
            lr_alz = 3e-5
        elif epoch <= 26:
            lr_alz = 1.5e-5
        elif epoch <= 29:
            lr_alz = 3e-5   # Re-warm burst: escape negative-momentum collapse basin
        else:
            lr_alz = 1e-5
        # Update existing optimizer LR each epoch
        for pg in optimizer_alz.param_groups:
            pg["lr"] = lr_alz

        # ep27: Reset ALZ Adam state — clears Phase-1 stale negative momentum
        # At ep27, beta1=0.9 means 0.9^17 ≈ 35% of Phase-1 negative momentum still active.
        # At lr=1e-5 (without reset), escaping the collapse basin takes ~10+ joint epochs.
        # Resetting m/v here + 3-epoch re-warm cuts that to ~2-3 epochs.
        if epoch == 27:
            optimizer_alz.state.clear()
            print("🔄 ep27: ALZ optimizer state reset — stale negative momentum cleared for joint phase")


        # Section F: PHASE-AWARE CURRICULUM (Sourced from top-level PHASE_CONFIG)
        active_phase = None
        for cutoff in sorted(PHASE_CONFIG.keys()):
            if epoch <= cutoff:
                active_phase = PHASE_CONFIG[cutoff]
                break
        
        if active_phase is None: 
            active_phase = (True, True) # Fallback to co-training
            
        TRAIN_ALZ, TRAIN_SEG = active_phase
            
        # Hard isolation via grad-freezing
        for p in model.encoder.parameters():
            p.requires_grad = TRAIN_SEG
        for p in model.tumor_decoder.parameters(): # Specific to tumor_decoder
            p.requires_grad = TRAIN_SEG
        for p in model.stroke_decoder.parameters(): # Specific to stroke_decoder
            p.requires_grad = TRAIN_SEG
        for p in model.tumor_presence.parameters(): # Specific to tumor_presence
            p.requires_grad = TRAIN_SEG
        for p in model.stroke_presence.parameters(): # Specific to stroke_presence
            p.requires_grad = TRAIN_SEG
        for p in model.bottleneck.parameters(): # Specific to bottleneck
            p.requires_grad = TRAIN_SEG
        for p in model.decision_head.parameters(): # Specific to decision_head
            p.requires_grad = TRAIN_SEG
        for p in model.alz_encoder.parameters():
            p.requires_grad = TRAIN_ALZ
            
        LAMBDA_CLS = 1.0  # Alzheimer task scaling (Part C)

        # Fix 6/7: Temperature freeze aligned to Phase 3 start (epoch 27)
        model.temperature.requires_grad = (epoch >= 27)

        # Fix 6: Uncertainty / decision head staged to Phase 3 (epoch 27+)
        use_uncertainty = (epoch >= 27)
        use_decision    = (epoch >= 27)
        if 27 <= epoch <= 31:
            ep_decision_weight = 0.3   # warm-in decision head gently
        else:
            ep_decision_weight = 1.0

        # Loop Setup
        model.train()

        
        tumor_iter  = iter(tumor_loader)  if (TRAIN_SEG and tumor_loader) else None
        stroke_iter = iter(stroke_loader) if (TRAIN_SEG and stroke_loader) else None
        alz_iter    = iter(alz_loader)    if (TRAIN_ALZ and alz_loader) else None
        
        t_losses, s_losses, a_losses = [], [], []
        t_et_dices, t_ncr_dices, t_ed_dices = [], [], []
        s_dices = []
        
        avg_et, avg_ncr, avg_ed, avg_tumor_mean, avg_stroke, avg_alz_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # Fix #9: Set current epoch on model for smooth transformer blend
        model._current_epoch = epoch
        if epoch == 7:
            print("  ⚡ Transformer bottleneck at full alpha (epoch 7)")

        # E1: Step cap for SEG-only phase (Kaggle time budget)
        if TRAIN_SEG:
            seg_lens = [len(l) for l in [tumor_loader, stroke_loader] if l is not None]
            steps_seg = max(seg_lens) if seg_lens else 0
            if TRAIN_SEG and not TRAIN_ALZ:
                steps_seg = min(steps_seg, 1000)
        else:
            steps_seg = 0
        steps_alz = len(alz_loader) if (TRAIN_ALZ and alz_loader) else 0
        max_steps = max(steps_alz, steps_seg)

        print(f"\n{'='*75}")
        print(f"  EPOCH {epoch}/{EPOCHS} | TRAIN_ALZ={TRAIN_ALZ} TRAIN_SEG={TRAIN_SEG} AMP={USE_AMP}")
        print(f"{'='*75}")
        
        if TRAIN_ALZ: optimizer_alz.zero_grad()
        if TRAIN_SEG: optimizer_shared.zero_grad()

        for step in range(max_steps):            
            if TRAIN_ALZ and alz_loader:
                try: batch = next(alz_iter)
                except (StopIteration, TypeError): alz_iter = iter(alz_loader); batch = next(alz_iter)
                
                loss = train_step(model, batch, "alzheimer", optimizer_alz, scaler_alz, epoch,
                                  use_uncertainty=use_uncertainty, use_decision=use_decision,
                                  ep_decision_weight=ep_decision_weight)
                if loss is not None:
                    loss = loss / ACCUM_STEPS
                    scaler_alz.scale(loss).backward()
                    
                    if (step + 1) % ACCUM_STEPS == 0 or step == max_steps - 1:
                        scaler_alz.unscale_(optimizer_alz)
                        # Fix 2: Adaptive gradient clipping (looser early for feature formation)
                        alz_clip = 0.7 if epoch <= 6 else 0.5
                        torch.nn.utils.clip_grad_norm_(model.alz_encoder.parameters(), alz_clip)
                        # Fix 4: Removed unscientific gradient noise floor — destroys convergence quality
                        scaler_alz.step(optimizer_alz)
                        scaler_alz.update()
                        optimizer_alz.zero_grad()
                        
                    a_losses.append(loss.item() * ACCUM_STEPS)

            if TRAIN_SEG:
                # Ensure batch and loss are reset every step to avoid stale metrics
                loss_t, loss_s = None, None
                t_batch, s_batch = None, None
                
                # Part 6: Balancing Updates — ALZ every step, SEG every 2 steps in Joint Phase
                # This prevents heavy decoder updates from over-dominating shared encoder gradients.
                skip_seg = TRAIN_ALZ and (step % 2 != 0)
                
                if not skip_seg:
                    # Change 12: Independent task guards for dataset robustness
                    if tumor_loader:
                        try: t_batch = next(tumor_iter)
                        except (StopIteration, TypeError, AttributeError): tumor_iter = iter(tumor_loader); t_batch = next(tumor_iter)
                        loss_t = train_step(model, t_batch, "tumor", optimizer_shared, scaler_shared, epoch,
                                            use_uncertainty=use_uncertainty, use_decision=use_decision,
                                            ep_decision_weight=ep_decision_weight)
                    
                    if stroke_loader:
                        try: s_batch = next(stroke_iter)
                        except (StopIteration, TypeError, AttributeError): stroke_iter = iter(stroke_loader); s_batch = next(stroke_iter)
                        loss_s = train_step(model, s_batch, "stroke", optimizer_shared, scaler_shared, epoch,
                                            use_uncertainty=use_uncertainty, use_decision=use_decision,
                                            ep_decision_weight=ep_decision_weight)
                    
                    losses_to_combine = [l for l in [loss_t, loss_s] if l is not None]
                    if losses_to_combine:
                        total_loss = sum(losses_to_combine) / ACCUM_STEPS
                        scaler_shared.scale(total_loss).backward()
                        
                        if (step + 1) % ACCUM_STEPS == 0 or step == max_steps - 1:
                            scaler_shared.unscale_(optimizer_shared)
                            shared_params = optimizer_shared.param_groups[0]["params"] + optimizer_shared.param_groups[1]["params"]
                            # A3: Clip to 0.5 globally
                            torch.nn.utils.clip_grad_norm_(shared_params, 0.5)
                            if all(p.grad is None or torch.isfinite(p.grad).all() for p in shared_params):
                                scaler_shared.step(optimizer_shared)
                            scaler_shared.update()
                            optimizer_shared.zero_grad()
                        
                        if loss_t is not None: t_losses.append(loss_t.item())
                        if loss_s is not None: s_losses.append(loss_s.item())
                
                # Training Metrics (Every 10 steps, with batch guards)
                if step % 10 == 0:
                    with torch.no_grad():
                        if t_batch is not None:
                            t_res = model(t_batch["image"].to(DEVICE), active_seg=["tumor"])
                            t_probs = torch.sigmoid(t_res["segmentations"]["tumor"])
                            t_et_dices.append(quick_dice(t_probs[:, 0:1], t_batch["seg"][:, 0:1].to(DEVICE)))
                            t_ncr_dices.append(quick_dice(t_probs[:, 1:2], t_batch["seg"][:, 1:2].to(DEVICE)))
                            t_ed_dices.append(quick_dice(t_probs[:, 2:3], t_batch["seg"][:, 2:3].to(DEVICE)))

                        if s_batch is not None:
                            s_res = model(s_batch["image"].to(DEVICE), active_seg=["stroke"])
                            s_d = quick_dice(torch.sigmoid(s_res["segmentations"]["stroke"]), s_batch["seg"].to(DEVICE))
                            s_dices.append(s_d)
            
            # Print progress every 50 steps
            if (step + 1) % 50 == 0 or step == max_steps - 1:
                log_parts = [f"  Step {step+1:>4}/{max_steps}"]
                t_l = np.mean(t_losses[-50:]) if t_losses else 0.0
                s_l = np.mean(s_losses[-50:]) if s_losses else 0.0
                et_d = np.mean(t_et_dices[-50:]) if t_et_dices else 0.0
                s_d  = np.mean(s_dices[-50:]) if s_dices else 0.0
                log_parts.append(f"T_L={t_l:.3f} | S_L={s_l:.3f} | ET_D={et_d:.3f} | Str_D={s_d:.3f}")
                log_parts.append(f"A_L={np.mean(a_losses[-50:]):.3f}" if a_losses else "A_L=n/a")
                tqdm.write(" | ".join(log_parts))

        # Section J: LOGGING SYSTEM
        print(f"\n🏁 EPOCH {epoch} SUMMARY")
        if TRAIN_ALZ:
            avg_alz_loss = np.mean(a_losses) if a_losses else 0.0
            print(f"  🩺 Alzheimer | Loss: {avg_alz_loss:.4f}")
            with torch.no_grad():
                batch = next(iter(alz_val_loader))
                res = model(batch["image"].to(DEVICE), active_presence=["alzheimer"])
                # res['presence']['alzheimer'] is a single (B,1) tensor after temp scaling
                logit = res["presence"]["alzheimer"]
                # Fix #12: Remove batch-size-1 std (always 0.0, misleading)
                # Real multi-sample std is in the B1+B2 diagnostic block below
                print(f"     Logit Mean: {logit.mean().item():.3f}")
                # FIX 4: Prediction monitoring (Diversity Hardening)
                prob_m = torch.sigmoid(logit).mean().item()
                print(f"     Pred prob mean: {prob_m:.3f}")

        if TRAIN_SEG:
            print(f"  🧠 Segmentation Summary (Training Samples)")
            # Log empty preds % and predicted volume (Section J)
            if s_dices:
                print(f"     Stroke | Empty Predictions: {sum(s == 0 for s in s_dices)/len(s_dices):.1%}")
                print(f"     Stroke | Mean Dice (Fixed 0.5): {np.mean(s_dices):.4f}")
            if t_et_dices:
                # Fix 8: Metric labeling honesty (Soft Dice vs. Val Hard Dice)
                print(f"     Tumor  | ET Soft Dice: {np.mean(t_et_dices):.4f} | WT Soft Dice: {np.mean(t_ed_dices):.4f}")

        # ISSUE 5 FIX: VRAM Edge Guard — clear cache after training loop
        # Releases stray tensors before validation HD95 which requires heavy memory.
        torch.cuda.empty_cache()
        sys.stdout.flush()

        # ─── Validation & Metrics History ──────────────────────────────────
        if TRAIN_ALZ:
            # Alzheimer Validation — full metrics (AUC + Brier + ECE)
            alz_metrics = evaluate_alzheimer_full(model, alz_val_loader)
            
            # Part 5: Sanity heuristic — detect weak learning signal early
            if epoch >= 5 and alz_metrics["auc"] < 0.55:
                print("  ⚠️ WARNING: Weak learning signal detected (val_auc < 0.55)")
            
            # Change 7: Step Scheduler (AUC-driven)
            if scheduler_alz is not None:
                scheduler_alz.step(alz_metrics["auc"])
                print(f"     Alz LR: {optimizer_alz.param_groups[0]['lr']:.2e}")

            m = metrics_history["val"]["alz"]
            m["auc"].append(round(alz_metrics["auc"], 4))
            m["auprc"].append(round(alz_metrics["auprc"], 4))
            m["brier"].append(round(alz_metrics["brier"], 4))
            m["ece"].append(round(alz_metrics["ece"], 4))
            m["f1"].append(round(alz_metrics["f1"], 4))
            m["accuracy"].append(round(alz_metrics["accuracy"], 4))
            m["precision"].append(round(alz_metrics["precision"], 4))
            m["recall"].append(round(alz_metrics["recall"], 4))
        else:
            # Append None when Alzheimer not trained this epoch
            for key in ["auc", "auprc", "brier", "ece", "f1", "accuracy", "precision", "recall"]:
                metrics_history["val"]["alz"][key].append(None)

        if TRAIN_SEG:
            seg_metrics = evaluate_segmentation_full(
                model, tumor_val_loader, stroke_val_loader, epoch
            )
            run_hd = (epoch % EVAL_HD_EVERY == 0)

            tv = metrics_history["val"]["tumor"]
            sv = metrics_history["val"]["stroke"]

            def _r(v): return round(v, 4) if (v is not None and not np.isnan(v)) else None

            # Overlap — always available
            tv["et_dice"].append(_r(seg_metrics["tumor_et_dice"]))
            tv["tc_dice"].append(_r(seg_metrics["tumor_tc_dice"]))
            tv["wt_dice"].append(_r(seg_metrics["tumor_wt_dice"]))
            tv["tc_iou"].append(_r(seg_metrics["tumor_tc_iou"]))
            tv["wt_iou"].append(_r(seg_metrics["tumor_wt_iou"]))
            tv["tc_ve"].append(_r(seg_metrics["tumor_tc_ve"]))
            tv["wt_ve"].append(_r(seg_metrics["tumor_wt_ve"]))
            sv["dice"].append(_r(seg_metrics["stroke_dice"]))
            sv["iou"].append(_r(seg_metrics["stroke_iou"]))
            sv["ve"].append(_r(seg_metrics["stroke_ve"]))

            # Boundary (HD95/ASD) — None when not computed this epoch
            tv["et_hd95"].append(_r(seg_metrics["tumor_et_hd95"]) if run_hd else None)
            tv["wt_hd95"].append(_r(seg_metrics["tumor_wt_hd95"]) if run_hd else None)
            tv["et_asd"].append(_r(seg_metrics["tumor_et_asd"])   if run_hd else None)
            tv["wt_asd"].append(_r(seg_metrics["tumor_wt_asd"])   if run_hd else None)
            sv["hd95"].append(_r(seg_metrics["stroke_hd95"]) if run_hd else None)
            sv["asd"].append(_r(seg_metrics["stroke_asd"])   if run_hd else None)
        else:
            # Append None for all seg val metrics when SEG not trained
            for key in ["et_dice", "tc_dice", "wt_dice",
                        "tc_iou", "wt_iou", "et_hd95", "wt_hd95",
                        "et_asd",  "wt_asd",  "tc_ve",   "wt_ve"]:
                metrics_history["val"]["tumor"][key].append(None)
            for key in ["dice", "iou", "hd95", "asd", "ve"]:
                metrics_history["val"]["stroke"][key].append(None)

        # ── Train metrics (Fix 10: Compute actual averages) ───────────────
        avg_et = np.mean(t_et_dices) if t_et_dices else 0.0
        avg_tumor_mean = (np.mean(t_et_dices) + np.mean(t_ncr_dices) + np.mean(t_ed_dices))/3 if t_et_dices else 0.0
        avg_stroke = np.mean(s_dices) if s_dices else 0.0
        
        metrics_history["epoch"].append(epoch)
        metrics_history["train"]["tumor"]["et_dice"].append(round(avg_et, 4))
        metrics_history["train"]["tumor"]["mean_dice"].append(round(avg_tumor_mean, 4))
        metrics_history["train"]["tumor"]["loss"].append(round(float(np.mean(t_losses)) if t_losses else 0.0, 4))
        metrics_history["train"]["stroke"]["dice"].append(round(avg_stroke, 4))
        metrics_history["train"]["stroke"]["loss"].append(round(float(np.mean(s_losses)) if s_losses else 0.0, 4))
        metrics_history["train"]["alz"]["loss"].append(round(float(np.mean(a_losses)) if a_losses else 0.0, 4))

        # ── Global composite score ───────────────────────────────────────────────
        # Score = wt_dice + stroke_dice + (alz_auc - alz_brier) - hd_penalty
        # During SEG-only phase (ep16-30), alz metrics are None.
        # Carry forward both AUC and Brier from last Alzheimer validation epoch
        # so the score stays meaningful and consistent across all phases.
        wt_dice_ep   = metrics_history["val"]["tumor"]["wt_dice"][-1] or 0.0
        s_dice_ep    = metrics_history["val"]["stroke"]["dice"][-1]    or 0.0

        _raw_auc   = metrics_history["val"]["alz"]["auc"][-1]
        _raw_brier = metrics_history["val"]["alz"]["brier"][-1]

        # Carry forward last known values for both metrics
        past_aucs   = [v for v in metrics_history["val"]["alz"]["auc"]   if v is not None]
        past_briers = [v for v in metrics_history["val"]["alz"]["brier"] if v is not None]
        alz_auc_ep   = _raw_auc   if _raw_auc   is not None else (past_aucs[-1]   if past_aucs   else 0.5)
        alz_brier_ep = _raw_brier if _raw_brier is not None else (past_briers[-1] if past_briers else 0.25)

        wt_hd95_ep = metrics_history["val"]["tumor"]["wt_hd95"][-1]
        # Soft HD95 penalty: cap at 50mm, scale 0.005 → max deduction 0.25
        hd_penalty = 0.005 * min(wt_hd95_ep, 50.0) if wt_hd95_ep is not None else 0.0

        # Calibration-aware: AUC - Brier rewards models that are both accurate AND calibrated
        global_score = round(wt_dice_ep + s_dice_ep + (alz_auc_ep - alz_brier_ep) - hd_penalty, 4)
        metrics_history["meta"]["score"].append(global_score)

        if metrics_history["meta"]["best_score"] is None or global_score > metrics_history["meta"]["best_score"]:
            metrics_history["meta"]["best_score"] = global_score
            metrics_history["meta"]["best_epoch"] = epoch
            print(f"⭐ New best global score: {global_score:.4f} at epoch {epoch}")

        print(f"     LR Shared: {optimizer_shared.param_groups[0]['lr']:.2e} | LR Alz: {optimizer_alz.param_groups[0]['lr']:.2e}")
        
        sys.stdout.flush()
        
        # ── Checkpoint: full resume state (FIX 2)
        torch.save({
            "model":    model.state_dict(),
            "optimizer_shared": optimizer_shared.state_dict(),
            "optimizer_alz":    optimizer_alz.state_dict(),
            "scaler_shared":    scaler_shared.state_dict(),
            "scaler_alz":       scaler_alz.state_dict(),
            "epoch":    epoch,
            "metrics":  metrics_history,
        }, CHECKPOINT_DIR / "neurox_last.pth")
        
        # ── B1 + B2: Research-Grade Alzheimer Diagnostics ─────────────────
        if TRAIN_ALZ:
            model.eval()
            with torch.no_grad():
                logits_all = []
                feats_all  = []
                for b_i, b in enumerate(alz_val_loader):
                    if b_i >= 30: break  # B2: 30-sample logit distribution
                    img_c = b["image"].to(DEVICE)
                    logit_v, _ = model.alz_encoder(img_c)
                    logits_all.append(logit_v.cpu())
                    if b_i < 20:         # B1: 20-sample feature variance
                        feat_v = model.alz_encoder.extract_features(img_c)
                        feats_all.append(feat_v.cpu())
                
                if logits_all:
                    all_l  = torch.cat(logits_all)
                    l_std  = all_l.std().item()
                    logit_status = "✅ healthy" if l_std > 0.05 else "🚨 COLLAPSED"
                    print(f"  🔍 Logit Std  (30 val): {l_std:.4f} ({logit_status})")
                    # Part 3: Learning Enforcement — exit if model collapses after warmup
                    if l_std < 0.05 and epoch >= 3:
                        print(f"❌ NOT LEARNING (logit_std={l_std:.4f}) — STOP RUN to prevent wasted time")
                        break # Exit epoch loop
                
                if feats_all:
                    all_f     = torch.cat(feats_all)
                    feat_std  = all_f.std().item()
                    feat_status = "✅ healthy" if feat_std > 0.1 else "🚨 COLLAPSED"
                    print(f"  🔍 Feature Std (20 val): {feat_std:.4f} ({feat_status})")
            model.train()
        
        # ── Checkpoint: inference weights + structured metrics
        torch.save({
            "model_state": model.state_dict(),
            "metrics":     metrics_history,
        }, CHECKPOINT_DIR / "neurox_model.pth")

        # Fix 7: JSON export safety
        try:
            metrics_json_path = CHECKPOINT_DIR / "metrics.json"
            with open(metrics_json_path, "w") as _jf:
                json.dump(metrics_history, _jf, default=lambda x: None if x != x else x)
        except Exception as _je:
            print(f"⚠️ JSON export failed (non-critical): {_je}")
        
        print(f"✅ Epoch {epoch} | Score={metrics_history['meta']['score'][-1]:.4f} "
              f"(Best: {metrics_history['meta']['best_score']:.4f} @ ep{metrics_history['meta']['best_epoch']}). "
              f"Checkpoints saved.")

        # (Cache flushing disabled natively: preventing massive re-computation pipeline degradation)

    # ── Post-Training: Final Temperature Calibration ────────────────────────
    # Optimise temperature on Alzheimer val set so inference probabilities are calibrated
    if alz_val_loader is not None:
        calibrate_model(model, alz_val_loader)
        # Re-save neurox_model.pth with calibrated temperature included
        torch.save({
            "model_state": model.state_dict(),
            "metrics":     metrics_history,
        }, CHECKPOINT_DIR / "neurox_model.pth")
        print("✅ Calibrated model saved to neurox_model.pth")


# ═══════════════════════════════════════════════════════════════════════════
#  CALIBRATION (runs after final epoch, optimises temperature on val set)
# ═══════════════════════════════════════════════════════════════════════════


def calibrate_model(model, val_loader):
    """Optimise temperature on the Alzheimer validation set after training.
    
    Called at end of main() so temperature is correctly calibrated for inference.
    Uses raw (uncalibrated) logits as input to Adam — temperature is the param.
    """
    print("\n🔥 Optimizing Temperature Scaling on Validation...")
    model.eval()
    
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            img = batch["image"].to(DEVICE)
            logit, _ = model.alz_encoder(img)
            all_logits.append(logit)
            all_labels.append(batch["presence"]["alzheimer"].to(DEVICE))
            
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    # ✅ Fix 3 — optional safety guard
    if torch.isnan(all_logits).any():
        print("❌ NaN logits — skipping calibration")
        return
    
    # Audit Fix 2: Safety guard for empty validation cohorts
    if all_logits.numel() == 0:
        print("   ⚠️ Calibration skipped: No validation samples available.")
        return
    
    # Ensure temperature requires grad for optimization
    model.temperature.requires_grad = True
    
    # ✅ Fix 2 — disable LBFGS (recommended)
    t_optimizer = torch.optim.Adam([model.temperature], lr=1e-2)
    
    # Smoothing Alignment: prevents temperature over-shrinking logits due to training smoothing
    target_smoothed = all_labels * 0.9 + 0.05
    
    for _ in range(100):
        t_optimizer.zero_grad()
        # ✅ Fix 1 — clamp temperature in calibration
        temp = model.temperature.clamp(0.01, 10.0)
        loss = F.binary_cross_entropy_with_logits(all_logits / temp, target_smoothed)
        loss.backward()
        t_optimizer.step()
        
    print(f"✅ Final Temperature: {model.temperature.item():.4f}")
    model.train()

if __name__ == "__main__":
    main()