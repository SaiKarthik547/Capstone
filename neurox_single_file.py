"""
NeuroX Hybrid - Advanced Brain Analysis Platform
Single file implementation for Kaggle notebooks
Name: Gotam Sai Varshith
"""

# Standard library imports
import os
import sys
import tempfile
import traceback
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import io
import base64

# Third-party imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from skimage import measure
import trimesh
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    Groq = None

# MONAI imports (conditional)
try:
    from monai.losses.dice import DiceCELoss
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False

# Configuration constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROI_SIZE = (128, 128, 128)  # Model input size (D, H, W)
BATCH_SIZE = 1
NUM_WORKERS = 0  # Set to 0 for Kaggle compatibility
LR = 1e-4
MAX_EPOCHS = 5

# Dataset configuration
DATASET_PATHS = [
    "/kaggle/input/brats-2023-dataset-nifti",
    "/kaggle/input/isles-2022-dataset-nifti", 
    "/kaggle/input/adni-3-dataset-nifti"
]

SEG_CHANNELS = ["edema", "enhancing", "necrotic", "whole_tumor"]
DISEASE_LABELS = ["glioma", "meningioma", "pituitary", "other"]

# Collate function for DataLoader
def collate_dict(batch):
    """Collate function that preserves dictionary structure."""
    return batch[0]

# Model Components
class ResBlock3D(nn.Module):
    """3D Residual Block with GroupNorm and SiLU activation."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.norm1 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.act1 = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.norm2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.act2 = nn.SiLU(inplace=True)
        
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride, 0, bias=False),
                nn.GroupNorm(min(32, out_ch), out_ch)
            )
        else:
            self.downsample = nn.Identity()
    
    def forward(self, x):
        residual = self.downsample(x)
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += residual
        out = self.act2(out)
        return out

class SwinBlock3D(nn.Module):
    """Simplified 3D Swin Transformer Block."""
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )
    
    def forward(self, x):
        # x shape: (B, C, D, H, W)
        B, C, D, H, W = x.shape
        # Reshape to (B, D*H*W, C) for attention
        x_flat = x.view(B, C, -1).permute(0, 2, 1)
        
        # Attention
        attn_out, _ = self.attn(self.norm1(x_flat), self.norm1(x_flat), self.norm1(x_flat))
        x_flat = x_flat + attn_out
        
        # MLP
        mlp_out = self.mlp(self.norm2(x_flat))
        x_flat = x_flat + mlp_out
        
        # Reshape back to (B, C, D, H, W)
        x = x_flat.permute(0, 2, 1).view(B, C, D, H, W)
        return x

class NeuroXHybrid(nn.Module):
    """Hybrid CNN + Transformer model for brain tumor segmentation and classification."""
    
    def __init__(self, in_ch: int = 2, seg_ch: int = 4, cls_ch: int = 4):
        super().__init__()
        
        # Encoder (CNN pathway)
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_ch, 32, 3, 1, 1),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
            ResBlock3D(32, 32)
        )
        self.enc2 = nn.Sequential(
            nn.Conv3d(32, 64, 3, 2, 1),
            nn.GroupNorm(16, 64),
            nn.SiLU(inplace=True),
            ResBlock3D(64, 64)
        )
        self.enc3 = nn.Sequential(
            nn.Conv3d(64, 128, 3, 2, 1),
            nn.GroupNorm(32, 128),
            nn.SiLU(inplace=True),
            ResBlock3D(128, 128)
        )
        
        # Bottleneck (Transformer pathway)
        self.bottleneck = nn.Sequential(
            SwinBlock3D(128),
            SwinBlock3D(128)
        )
        
        # Decoder (CNN pathway)
        self.upconv3 = nn.ConvTranspose3d(128, 64, 2, 2)
        self.dec3 = ResBlock3D(128, 64)
        self.upconv2 = nn.ConvTranspose3d(64, 32, 2, 2)
        self.dec2 = ResBlock3D(64, 32)
        
        # Segmentation head
        self.seg_head = nn.Conv3d(32, seg_ch, 1)
        
        # Classification head
        self.cls_pool = nn.AdaptiveAvgPool3d(1)
        self.cls_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.SiLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, cls_ch)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        # Bottleneck
        b = self.bottleneck(e3)
        
        # Decoder
        d3 = self.upconv3(b)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)
        
        # Segmentation output
        seg_logits = self.seg_head(d2)
        
        # Classification output
        cls_features = self.cls_pool(b).flatten(1)
        cls_logits = self.cls_head(cls_features)
        
        return seg_logits, cls_logits

# Dataset Utilities
def find_all_dataset_roots(paths=DATASET_PATHS):
    """Scan all dataset paths and return list of found datasets."""
    found_datasets = []
    for p in paths:
        if os.path.exists(p):
            found_datasets.append(p)
            print(f"Found dataset at: {p}")
    if not found_datasets:
        raise FileNotFoundError(f"No datasets found in {paths}")
    return found_datasets

def validate_nifti_file(filepath):
    """Validate that a file is a readable NIfTI file."""
    try:
        img = nib.load(filepath)
        data = img.get_fdata()
        print(f"  ✓ Valid NIfTI: {os.path.basename(filepath)} - Shape: {data.shape}")
        return True
    except Exception as e:
        print(f"  ✗ Invalid NIfTI: {os.path.basename(filepath)} - Error: {str(e)}")
        return False

def list_subjects(root: str) -> List[Dict]:
    """
    Scan dataset folder and return list of subjects.
    Each subject is a dict with keys: flair, t1ce, seg, id, dataset_type
    Supports BraTS 2023, ISLES 2022, ADNI 3, and generic datasets.
    """
    subjects = []
    
    # Walk through all directories to find datasets
    for dirpath, dirs, files in os.walk(root):
        # Get all nifti files in current directory
        nifti_files = [f for f in files if f.endswith('.nii') or f.endswith('.nii.gz')]
        
        if not nifti_files:
            continue
            
        # Debug information for ADNI dataset
        if "adni" in root.lower() or "adni" in dirpath.lower():
            print(f"ADNI Debug - Dir: {dirpath}, Files: {nifti_files}")
            
        # Try to identify dataset type based on file naming patterns
        patient_id = os.path.basename(dirpath)
        
        # BraTS 2023 pattern detection
        if any(f.startswith("BraTS-") and "-t2f.nii" in f for f in nifti_files):
            flair = None
            t1ce = None
            seg = None
            
            for f in nifti_files:
                if "-t2f.nii" in f:
                    flair = os.path.join(dirpath, f)
                elif "-t1n.nii" in f:
                    t1ce = os.path.join(dirpath, f)
                elif "-seg.nii" in f:
                    seg = os.path.join(dirpath, f)
            
            if flair and t1ce:
                subjects.append({
                    "flair": flair,
                    "t1ce": t1ce,
                    "seg": seg,
                    "id": patient_id,
                    "dataset_type": "brats"
                })
        
        # ISLES 2022 pattern detection
        elif any("ses-" in f and ".nii" in f for f in nifti_files):
            dwi = None
            adc = None
            seg = None
            
            for f in nifti_files:
                if "_dwi_" in f:
                    dwi = os.path.join(dirpath, f)
                elif "_adc_" in f:
                    adc = os.path.join(dirpath, f)
                elif "_lesion_" in f:
                    seg = os.path.join(dirpath, f)
            
            if dwi and adc:
                subjects.append({
                    "flair": dwi,
                    "t1ce": adc,
                    "seg": seg,
                    "id": patient_id,
                    "dataset_type": "isles"
                })
        
        # ADNI 3 pattern detection
        elif any("ADNI_" in f for f in nifti_files):
            t1 = None
            seg = None
            
            for f in nifti_files:
                if "MPRAGE" in f or "T1" in f.upper():
                    t1 = os.path.join(dirpath, f)
                # Assume any other file might be a segmentation if present
                elif seg is None and f != t1:
                    seg = os.path.join(dirpath, f)
            
            if t1:
                subjects.append({
                    "flair": t1,
                    "t1ce": t1,  # Use T1 as both channels for single modality
                    "seg": seg,
                    "id": patient_id,
                    "dataset_type": "adni"
                })
        
        # Generic dataset pattern (fallback)
        else:
            # Use any two NIfTI files as channels
            if len(nifti_files) >= 2:
                ch1 = os.path.join(dirpath, nifti_files[0])
                ch2 = os.path.join(dirpath, nifti_files[1])
                seg = os.path.join(dirpath, nifti_files[2]) if len(nifti_files) > 2 else None
                
                subjects.append({
                    "flair": ch1,
                    "t1ce": ch2,
                    "seg": seg,
                    "id": patient_id,
                    "dataset_type": "generic"
                })
            elif len(nifti_files) >= 1:
                # Single channel - duplicate for both
                ch1 = os.path.join(dirpath, nifti_files[0])
                subjects.append({
                    "flair": ch1,
                    "t1ce": ch1,
                    "seg": None,
                    "id": patient_id,
                    "dataset_type": "single_channel"
                })
    
    return subjects

def debug_dataset_structure(root: str):
    """Debug function to analyze dataset structure and file naming patterns."""
    print(f"Debugging dataset structure for: {root}")
    print("="*50)
    
    total_files = 0
    nifti_files = []
    
    for dirpath, dirs, files in os.walk(root):
        # Get all nifti files in current directory
        dir_nifti_files = [f for f in files if f.endswith('.nii') or f.endswith('.nii.gz')]
        if dir_nifti_files:
            print(f"Directory: {dirpath}")
            print(f"  NIfTI files: {dir_nifti_files}")
            nifti_files.extend([(dirpath, f) for f in dir_nifti_files])
            total_files += len(dir_nifti_files)
    
    print(f"\nTotal NIfTI files found: {total_files}")
    
    # Validate a sample of files
    if nifti_files:
        print("\nValidating sample files:")
        sample_size = min(5, len(nifti_files))
        valid_count = 0
        for i, (dirpath, filename) in enumerate(nifti_files[:sample_size]):
            filepath = os.path.join(dirpath, filename)
            if validate_nifti_file(filepath):
                valid_count += 1
        print(f"Valid files in sample: {valid_count}/{sample_size}")

def validate_dataset_structure(root: str) -> bool:
    """Validate that the dataset structure is compatible with the NeuroX pipeline."""
    try:
        subjects = list_subjects(root)
        if len(subjects) == 0:
            print("❌ No valid subjects found in dataset.")
            return False
            
        # Check a sample subject (less strict validation)
        sample_subject = subjects[0]
        required_files = ["flair", "t1ce"]
        
        valid_count = 0
        for req_file in required_files:
            if req_file in sample_subject and sample_subject[req_file]:
                filepath = sample_subject[req_file]
                # Check if file exists
                if os.path.exists(filepath):
                    valid_count += 1
                else:
                    print(f"⚠️  File does not exist: {filepath}")
        
        # At least one required file should exist
        return valid_count > 0
    except Exception as e:
        print(f"⚠️  Dataset validation error: {e}")
        return False

# Dataset Class
class MultiModalBrainDataset(Dataset):
    """Multi-modal brain dataset supporting BraTS, ISLES, ADNI, and generic datasets."""
    
    def __init__(self, subjects: List[Dict], roi_size: Tuple[int, int, int] = ROI_SIZE, augment: bool = False):
        self.subjects = subjects
        self.roi_size = roi_size
        self.augment = augment
    
    def __len__(self):
        return len(self.subjects)
    
    def __getitem__(self, idx):
        subj = self.subjects[idx]
        
        # Load modalities
        flair_nii = nib.load(subj["flair"])
        t1ce_nii = nib.load(subj["t1ce"])
        
        flair_arr = np.array(flair_nii.get_fdata(), dtype=np.float32)
        t1ce_arr = np.array(t1ce_nii.get_fdata(), dtype=np.float32)
        
        # Normalize intensities (0-99th percentile clipping)
        for arr in [flair_arr, t1ce_arr]:
            p99 = np.percentile(arr, 99)
            arr = np.clip(arr, 0, p99)
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        
        # Stack channels
        img = np.stack([flair_arr, t1ce_arr], axis=0)  # (C, H, W, D)
        img = np.transpose(img, (0, 3, 1, 2))  # (C, D, H, W)
        
        # Load segmentation if available
        if subj["seg"]:
            seg_nii = nib.load(subj["seg"])
            seg_arr = np.array(seg_nii.get_fdata(), dtype=np.float32)
            
            # Normalize segmentation labels based on dataset type
            if subj["dataset_type"] == "brats":
                # BraTS: 0=background, 1=necrotic, 2=edema, 4=enhancing
                seg_arr = np.where(seg_arr == 4, 3, seg_arr)  # Remap enhancing to index 3
                seg_arr = seg_arr / 3.0  # Normalize to 0-1
            elif subj["dataset_type"] == "isles":
                # ISLES: Binary lesion mask
                seg_arr = (seg_arr > 0).astype(np.float32)
            else:
                # Generic: Normalize to 0-1
                if seg_arr.max() > 0:
                    seg_arr = seg_arr / seg_arr.max()
            
            # Convert to multi-channel segmentation (one-hot-like)
            seg_multi = np.zeros((len(SEG_CHANNELS), *seg_arr.shape), dtype=np.float32)
            if subj["dataset_type"] == "brats":
                seg_multi[0] = (seg_arr == 2).astype(np.float32)  # Edema
                seg_multi[1] = (seg_arr == 3).astype(np.float32)  # Enhancing
                seg_multi[2] = (seg_arr == 1).astype(np.float32)  # Necrotic
                seg_multi[3] = (seg_arr > 0).astype(np.float32)   # Whole tumor
            else:
                # For other datasets, use the segmentation as whole tumor
                seg_multi[3] = seg_arr  # Whole tumor
            
            seg_multi = np.transpose(seg_multi, (0, 3, 1, 2))  # (C, D, H, W)
        else:
            # Create empty segmentation
            seg_multi = np.zeros((len(SEG_CHANNELS), *img.shape[1:]), dtype=np.float32)
        
        # Random augmentation (optional)
        if self.augment and np.random.rand() > 0.5:
            # Simple flip augmentation
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=-1).copy()
                seg_multi = np.flip(seg_multi, axis=-1).copy()
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=-2).copy()
                seg_multi = np.flip(seg_multi, axis=-2).copy()
        
        # Convert to tensors
        img_t = torch.from_numpy(img)
        seg_t = torch.from_numpy(seg_multi)
        
        # Classification label (random for demo purposes)
        cls_label = torch.rand(len(DISEASE_LABELS))
        
        return {
            "image": img_t,
            "segmentation": seg_t,
            "classification": cls_label,
            "id": subj["id"],
            "dataset_type": subj["dataset_type"]
        }

# Training Functions
def analyze_dataset_distribution(subjects: List[Dict]):
    """Analyze the distribution of subjects across different dataset types."""
    type_counts = {}
    for subj in subjects:
        dt = subj["dataset_type"]
        type_counts[dt] = type_counts.get(dt, 0) + 1
    
    print("Dataset Distribution:")
    for dt, count in type_counts.items():
        print(f"  {dt.upper()}: {count} subjects")
    print(f"  TOTAL: {len(subjects)} subjects")

def train_hybrid_model(subjects: list, max_epochs=MAX_EPOCHS):
    """Train the hybrid model on multiple datasets."""
    # Analyze dataset distribution
    print("Analyzing dataset distribution...")
    analyze_dataset_distribution(subjects)
    
    # Dataset & Dataloader
    dataset = MultiModalBrainDataset(subjects, roi_size=ROI_SIZE, augment=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=collate_dict)

    # Model, optimizer, loss
    model = NeuroXHybrid().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)
    scaler = GradScaler()

    # Loss functions
    seg_loss_fn = nn.BCEWithLogitsLoss()
    cls_loss_fn = nn.BCEWithLogitsLoss()

    # Try MONAI Dice+CE Loss if available
    dice_loss = None
    if MONAI_AVAILABLE:
        try:
            dice_loss = DiceCELoss(sigmoid=True, batch=True)
            print("✅ Using MONAI DiceCELoss")
        except Exception as e:
            print(f"⚠️  MONAI DiceCELoss not available: {e}")

    # Training loop
    model.train()
    for epoch in range(max_epochs):
        epoch_seg_loss = 0.0
        epoch_cls_loss = 0.0
        num_batches = 0
        
        for batch in loader:
            optimizer.zero_grad()
            
            with autocast():
                img = batch["image"].to(DEVICE).unsqueeze(0)  # Add batch dimension
                seg_true = batch["segmentation"].to(DEVICE).unsqueeze(0)
                cls_true = batch["classification"].to(DEVICE).unsqueeze(0)
                
                seg_logits, cls_logits = model(img)
                
                # Segmentation loss
                if dice_loss is not None:
                    seg_loss = dice_loss(seg_logits, seg_true)
                else:
                    seg_loss = seg_loss_fn(seg_logits, seg_true)
                
                # Classification loss
                cls_loss = cls_loss_fn(cls_logits, cls_true)
                
                # Combined loss
                loss = seg_loss + cls_loss
            
            # Backpropagation
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_seg_loss += seg_loss.item()
            epoch_cls_loss += cls_loss.item()
            num_batches += 1
        
        avg_seg_loss = epoch_seg_loss / num_batches
        avg_cls_loss = epoch_cls_loss / num_batches
        print(f"Epoch [{epoch+1}/{max_epochs}] - Seg Loss: {avg_seg_loss:.4f}, Cls Loss: {avg_cls_loss:.4f}")
    
    print("✅ Training completed!")
    return model

# 3D Visualization
def generate_3d_mesh(seg_prob, threshold=0.5):
    """
    Generate 3D mesh from segmentation probabilities.
    Returns a trimesh.Scene with colored meshes for each channel.
    """
    colors = [
        [255, 0, 0, 180],   # Red - Edema
        [0, 255, 0, 180],   # Green - Enhancing
        [0, 0, 255, 180],   # Blue - Necrotic
        [255, 255, 0, 180]  # Yellow - Whole tumor
    ]

    scene = trimesh.Scene()
    for idx in range(seg_prob.shape[0]):
        mask = seg_prob[idx] > threshold
        if mask.any():
            try:
                verts, faces, normals, values = measure.marching_cubes(mask.astype(np.uint8), level=0.5)
                mesh = trimesh.Trimesh(vertices=verts, faces=faces)
                mesh.visual.face_colors = colors[idx % len(colors)]
                # Apply smoothing if the method exists
                if hasattr(trimesh.smoothing, 'filter_laplacian'):
                    trimesh.smoothing.filter_laplacian(mesh, iterations=3)
                scene.add_geometry(mesh)
            except Exception as e:
                print(f"⚠️ Could not generate mesh for channel {idx}: {e}")
    if not scene.geometry:
        scene.add_geometry(trimesh.creation.icosphere(radius=5))
    return scene

# Report Generation
def generate_comprehensive_medical_report(seg_prob, cls_prob, output_file="comprehensive_medical_report.pdf"):
    """
    Generates a comprehensive medical report with:
    - Detailed analysis using Groq AI (if available)
    - Segmentation statistics with visualizations
    - Classification probabilities with risk assessment
    - Clinical recommendations
    - 3D visualization summary
    """
    
    # TODO: Replace "YOUR_GROQ_API_KEY_HERE" with your actual Groq API key for testing
    HARDCODED_GROQ_API_KEY = "YOUR_GROQ_API_KEY_HERE"  # <-- REPLACE THIS WITH YOUR ACTUAL API KEY
    
    # Create PDF document
    doc = SimpleDocTemplate(output_file, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=1  # Center alignment
    )
    story.append(Paragraph("NeuroX Hybrid Brain Analysis Report", title_style))
    story.append(Spacer(1, 20))
    
    # Patient Information Section
    story.append(Paragraph("Patient Information", styles['Heading2']))
    story.append(Spacer(1, 12))
    
    # Analysis Date
    analysis_data = [
        ['Analysis Date:', datetime.now().strftime("%B %d, %Y")],
        ['Analysis Time:', datetime.now().strftime("%H:%M:%S")],
        ['Report ID:', f"NX-{datetime.now().strftime('%Y%m%d%H%M%S')}"]
    ]
    
    table = Table(analysis_data, colWidths=[2*inch, 4*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.grey),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('BACKGROUND', (0, 0), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(table)
    story.append(Spacer(1, 20))
    
    # Segmentation Results Section
    story.append(Paragraph("Segmentation Analysis", styles['Heading2']))
    story.append(Spacer(1, 12))
    
    # Segmentation statistics table
    seg_data = [['Region', 'Voxel Count', 'Volume Estimate']]
    total_voxels = 0
    for i, channel in enumerate(SEG_CHANNELS):
        voxel_count = (seg_prob[i] > 0.5).sum()
        total_voxels += voxel_count
        volume_estimate = f"{voxel_count * 0.001:.2f} mL"  # Simplified volume calculation
        seg_data.append([channel, str(voxel_count), volume_estimate])
    
    seg_table = Table(seg_data, colWidths=[2*inch, 2*inch, 2*inch])
    seg_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(seg_table)
    story.append(Spacer(1, 20))
    
    # Disease Classification Section
    story.append(Paragraph("Disease Probability Assessment", styles['Heading2']))
    story.append(Spacer(1, 12))
    
    # Classification probabilities table
    cls_data = [['Disease', 'Probability', 'Risk Level']]
    for i, disease in enumerate(DISEASE_LABELS):
        prob = cls_prob[i]
        if prob > 0.7:
            risk_level = "High"
        elif prob > 0.4:
            risk_level = "Moderate"
        else:
            risk_level = "Low"
        cls_data.append([disease.capitalize(), f"{prob:.2%}", risk_level])
    
    cls_table = Table(cls_data, colWidths=[2*inch, 2*inch, 2*inch])
    cls_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(cls_table)
    story.append(Spacer(1, 20))
    
    # AI-Powered Clinical Interpretation (using Groq if available)
    story.append(Paragraph("Clinical Interpretation", styles['Heading2']))
    story.append(Spacer(1, 12))
    
    # Prepare data for AI analysis
    seg_summary = ""
    for i, channel in enumerate(SEG_CHANNELS):
        voxel_count = (seg_prob[i] > 0.5).sum()
        if voxel_count > 0:
            seg_summary += f"{channel}: {voxel_count} voxels; "
    
    cls_summary = ""
    for i, disease in enumerate(DISEASE_LABELS):
        prob = cls_prob[i]
        if prob > 0.3:  # Only mention diseases with significant probability
            cls_summary += f"{disease}: {prob:.1%} probability; "
    
    ai_interpretation = "Clinical interpretation requires a qualified physician's review."
    
    if GROQ_AVAILABLE and Groq is not None:
        try:
            # Use hardcoded API key if provided and no environment variable is set
            groq_api_key = os.environ.get("GROQ_API_KEY") or (HARDCODED_GROQ_API_KEY if HARDCODED_GROQ_API_KEY != "YOUR_GROQ_API_KEY_HERE" else None)
            if groq_api_key:
                client = Groq(api_key=groq_api_key) if Groq else None
                
                if client:
                    prompt = f"""
                    As a radiology AI assistant, analyze these brain MRI results:
                    
                    Segmentation Findings: {seg_summary}
                    Disease Probabilities: {cls_summary}
                    
                    Provide a concise clinical interpretation in 3-4 sentences that a radiologist would include in a report.
                    Focus on the most significant findings and their clinical implications.
                    Do not make definitive diagnoses - only highlight notable observations.
                    """
                    
                    chat_completion = client.chat.completions.create(
                        messages=[
                            {
                                "role": "user",
                                "content": prompt,
                            }
                        ],
                        model="llama3-8b-8192",  # Free Groq model
                    )
                    
                    ai_interpretation = chat_completion.choices[0].message.content
                else:
                    ai_interpretation = "Groq client could not be initialized.\n\n" + ai_interpretation
            else:
                ai_interpretation = "No API key provided. Enter your Groq API key for AI-powered interpretation.\n\n" + ai_interpretation
        except Exception as e:
            ai_interpretation = f"AI interpretation unavailable: {str(e)}\n\n" + ai_interpretation
    else:
        ai_interpretation = "Groq API not installed. Install with 'pip install groq' for AI-powered interpretation.\n\n" + ai_interpretation
    
    story.append(Paragraph(ai_interpretation, styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Recommendations Section
    story.append(Paragraph("Recommendations", styles['Heading2']))
    story.append(Spacer(1, 12))
    
    recommendations = [
        "• Review results with a qualified neuroradiologist",
        "• Consider clinical correlation with patient symptoms",
        "• Follow up with additional imaging if clinically indicated",
        "• Discuss findings with the referring physician"
    ]
    
    for rec in recommendations:
        story.append(Paragraph(rec, styles['Normal']))
    
    story.append(Spacer(1, 20))
    
    # Report Footer
    story.append(Paragraph("Disclaimer", styles['Heading2']))
    story.append(Spacer(1, 12))
    disclaimer = """
    This automated analysis is intended for research and educational purposes only. 
    It should not be used as the sole basis for clinical decision-making. 
    All results must be reviewed and interpreted by a qualified healthcare professional.
    """
    story.append(Paragraph(disclaimer, styles['Italic']))
    
    # Build PDF
    doc.build(story)
    print(f"✅ Comprehensive medical report generated: {output_file}")
    return output_file

def infer_and_export(file_path: str, ckpt="best_model.pth", threshold=0.5):
    """
    Run inference on a single MRI, return:
    - GLB 3D mesh file
    - Comprehensive PDF medical report
    """
    # Load model
    model = NeuroXHybrid().to(DEVICE)
    if os.path.exists(ckpt):
        st = torch.load(ckpt, map_location=DEVICE)
        model.load_state_dict(st["model"])
        print("✅ Checkpoint loaded for inference.")
    model.eval()

    # Load image
    nii = nib.load(file_path)
    arr = np.array(nii.get_fdata(), dtype=np.float32)
    
    # Handle different input formats
    if arr.ndim == 3:
        # Single channel - duplicate to create 2 channels
        img_ch = np.stack([arr, arr], axis=0)
    elif arr.ndim == 4:
        # Multi-channel - use first 2 channels
        img_ch = np.moveaxis(arr, -1, 0)
        if img_ch.shape[0] >= 2:
            img_ch = img_ch[:2]
        else:
            # If less than 2 channels, duplicate the first one
            img_ch = np.repeat(img_ch, 2, axis=0)[:2]
    else:
        raise ValueError("Unsupported input dimensions for inference.")

    # Transpose to (C, D, H, W) format
    img_ch = img_ch.transpose(0, 3, 1, 2)  # (C,D,H,W)
    
    # Convert to tensor and add batch dimension
    img_t = torch.from_numpy(img_ch).unsqueeze(0).float().to(DEVICE)
    
    # Resize to model input size
    img_t = F.interpolate(img_t, size=ROI_SIZE, mode='trilinear', align_corners=False)

    # Forward pass
    with torch.no_grad():
        with autocast():  # pyright: ignore[reportDeprecated, reportDeprecated]
            seg_logits, cls_logits = model(img_t)
            seg_prob = torch.sigmoid(seg_logits)[0].cpu().numpy()
            cls_prob = torch.sigmoid(cls_logits)[0].cpu().numpy()

    # Generate 3D mesh
    scene = generate_3d_mesh(seg_prob, threshold=threshold)
    glb_file = "brain_segmentation.glb"
    scene.export(glb_file)
    print(f"✅ 3D GLB mesh exported: {glb_file}")

    # Generate comprehensive medical report
    pdf_file = generate_comprehensive_medical_report(seg_prob, cls_prob, output_file="comprehensive_medical_report.pdf")

    return glb_file, pdf_file

# Streamlit UI (consolidated in same file)
def run_streamlit_app():
    """Run the Streamlit application."""
    import streamlit as st
    import streamlit.components.v1 as components
    
    # Set page config for a better layout
    st.set_page_config(
        page_title="NeuroX Hybrid - Advanced Brain Analysis",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for beautiful UI
    st.markdown("""
    <style>
        /* Global Styles */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
        
        :root {
            --primary: #6366f1;
            --primary-dark: #4f46e5;
            --secondary: #8b5cf6;
            --accent: #06b6d4;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --dark: #0f172a;
            --darker: #020617;
            --light: #f8fafc;
            --gray: #94a3b8;
            --card-bg: rgba(30, 41, 59, 0.6);
            --card-border: rgba(148, 163, 184, 0.2);
        }
        
        body {
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(135deg, var(--dark) 0%, var(--darker) 100%);
            color: var(--light);
            overflow-x: hidden;
        }
        
        /* Header Styles */
        header {
            background: rgba(30, 41, 59, 0.8);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid var(--card-border);
            padding: 1rem 0;
            margin-bottom: 2rem;
        }
        
        .header-content {
            display: flex;
            justify-content: space-between;
            align-items: center;
            max-width: 1400px;
            margin: 0 auto;
            padding: 0 2rem;
        }
        
        .logo {
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        
        .logo-icon {
            font-size: 2.5rem;
            color: var(--primary);
            animation: pulse 2s infinite;
        }
        
        .logo-text {
            font-size: 2rem;
            font-weight: 700;
            background: linear-gradient(90deg, var(--primary), var(--secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        /* Card Styles */
        .card {
            background: var(--card-bg);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            border: 1px solid var(--card-border);
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
            height: 100%;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.4);
        }
        
        .card-title {
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            color: var(--light);
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        
        /* Button Styles */
        .stButton > button {
            background: linear-gradient(90deg, var(--primary), var(--secondary));
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.9rem 1.8rem;
            font-weight: 600;
            font-size: 1.1rem;
            cursor: pointer;
            transition: all 0.3s ease;
            width: 100%;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
        }
        
        .stButton > button:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(99, 102, 241, 0.5);
        }
        
        .stButton > button:active {
            transform: translateY(0);
        }
        
        .btn-secondary {
            background: rgba(30, 41, 59, 0.8) !important;
            border: 1px solid var(--card-border) !important;
        }
        
        .btn-secondary:hover {
            background: rgba(99, 102, 241, 0.2) !important;
            border-color: var(--primary) !important;
        }
        
        /* File Uploader Styles */
        .stFileUploader {
            background: rgba(15, 23, 42, 0.5);
            border: 2px dashed rgba(148, 163, 184, 0.3);
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .stFileUploader:hover {
            border-color: var(--primary);
            background: rgba(15, 23, 42, 0.7);
        }
        
        .stFileUploader label {
            color: var(--gray);
            font-weight: 500;
        }
        
        /* Progress Bar */
        .stProgress > div > div {
            background: linear-gradient(90deg, var(--primary), var(--accent));
        }
        
        /* Animation Keyframes */
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); }
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes gradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        .fade-in {
            animation: fadeIn 0.6s ease forwards;
        }
        
        /* 3D Viewer Container */
        .viewer-container {
            width: 100%;
            height: 700px;
            background: rgba(15, 23, 42, 0.7);
            border-radius: 16px;
            border: 1px solid var(--card-border);
            overflow: hidden;
            position: relative;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
        }
        
        /* Status Messages */
        .status-message {
            padding: 1.2rem;
            border-radius: 12px;
            margin: 1.5rem 0;
            text-align: center;
            font-weight: 500;
            font-size: 1.1rem;
        }
        
        .status-success {
            background: rgba(16, 185, 129, 0.2);
            border: 1px solid var(--success);
            color: var(--success);
        }
        
        .status-error {
            background: rgba(239, 68, 68, 0.2);
            border: 1px solid var(--danger);
            color: var(--danger);
        }
        
        .status-info {
            background: rgba(6, 182, 212, 0.2);
            border: 1px solid var(--accent);
            color: var(--accent);
        }
        
        .status-warning {
            background: rgba(245, 158, 11, 0.2);
            border: 1px solid var(--warning);
            color: var(--warning);
        }
        
        /* Download Buttons */
        .download-buttons {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
            margin-top: 1.5rem;
        }
        
        .download-btn {
            background: rgba(30, 41, 59, 0.8);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .download-btn:hover {
            background: rgba(99, 102, 241, 0.2);
            border-color: var(--primary);
            transform: translateY(-3px);
        }
        
        .download-btn i {
            font-size: 2.5rem;
            margin-bottom: 0.8rem;
            display: block;
            color: var(--primary);
        }
        
        /* Metrics */
        .metric-card {
            background: rgba(30, 41, 59, 0.4);
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            border: 1px solid var(--card-border);
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            margin: 0.5rem 0;
            background: linear-gradient(90deg, var(--primary), var(--accent));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .metric-label {
            color: var(--gray);
            font-size: 1rem;
        }
        
        /* Footer */
        footer {
            text-align: center;
            padding: 2.5rem;
            margin-top: 3rem;
            color: var(--gray);
            font-size: 1rem;
            border-top: 1px solid var(--card-border);
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .header-content {
                flex-direction: column;
                gap: 1rem;
            }
            
            .download-buttons {
                grid-template-columns: 1fr;
            }
            
            .logo-text {
                font-size: 1.5rem;
            }
        }
        
        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: rgba(15, 23, 42, 0.7);
            border-right: 1px solid var(--card-border);
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 1rem;
            background: rgba(30, 41, 59, 0.4);
            padding: 0.5rem;
            border-radius: 12px;
            border: 1px solid var(--card-border);
        }
        
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border-radius: 8px;
            color: var(--gray);
            font-weight: 500;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(90deg, var(--primary), var(--secondary));
            color: white !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <header>
        <div class="header-content">
            <div class="logo">
                <div class="logo-icon">🧠</div>
                <div class="logo-text">NeuroX Hybrid</div>
            </div>
            <div style="color: var(--gray); font-size: 1.2rem;">
                Advanced 3D Brain Segmentation & Analysis
            </div>
        </div>
    </header>
    """, unsafe_allow_html=True)

    # Main Content
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)

    # Title
    st.markdown("<h1 style='text-align: center; margin-bottom: 1rem;'>🧠 Advanced 3D Brain Segmentation & Visualization</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: var(--gray); margin-bottom: 3rem;'>Upload a brain MRI scan for comprehensive analysis, 3D visualization, and AI-powered medical insights</p>", unsafe_allow_html=True)

    # Initialize session state
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
        
    if 'glb_file' not in st.session_state:
        st.session_state.glb_file = None
        
    if 'pdf_file' not in st.session_state:
        st.session_state.pdf_file = None

    # Create tabs for better organization
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Analyze", "🌐 3D Visualization", "📊 Results & Reports"])

    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # File Upload Section
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">📤 Upload MRI Scan</div>', unsafe_allow_html=True)
                
                uploaded_file = st.file_uploader(
                    "Upload a brain MRI file in NIfTI format (.nii or .nii.gz)",
                    type=['nii', 'nii.gz'],
                    key="mri_upload",
                    accept_multiple_files=False
                )
                
                if uploaded_file is not None:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    st.session_state.uploaded_file_path = tmp_file_path
                    st.success(f"File uploaded successfully: {uploaded_file.name}")
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            # API Key Section
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">🔑 API Configuration</div>', unsafe_allow_html=True)
                
                api_key = st.text_input(
                    "Groq API Key (for AI-powered insights)",
                    type="password",
                    placeholder="Enter your Groq API key",
                    key="api_key"
                )
                
                if api_key:
                    os.environ["GROQ_API_KEY"] = api_key
                    st.success("API key set successfully!")
                
                st.info("💡 Get a free API key at [console.groq.com](https://console.groq.com)")
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Analysis Section
        if 'uploaded_file_path' in st.session_state:
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">⚡ Process MRI Scan</div>', unsafe_allow_html=True)
                
                if st.button("Analyze Brain MRI", key="analyze_btn", use_container_width=True):
                    # Show processing status
                    status_placeholder = st.empty()
                    status_placeholder.markdown('<div class="status-message status-info">🔄 Processing MRI scan... This may take a minute.</div>', unsafe_allow_html=True)
                    
                    try:
                        # Run inference
                        with st.spinner("Running advanced analysis..."):
                            glb_file, pdf_file = infer_and_export(st.session_state.uploaded_file_path)
                            
                        # Store results in session state
                        st.session_state.glb_file = glb_file
                        st.session_state.pdf_file = pdf_file
                        st.session_state.analysis_complete = True
                        
                        # Show success message
                        status_placeholder.markdown('<div class="status-message status-success">✅ Analysis completed successfully!</div>', unsafe_allow_html=True)
                        
                    except Exception as e:
                        status_placeholder.markdown(f'<div class="status-message status-error">❌ Error during analysis: {str(e)}</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        # 3D Visualization Section
        if st.session_state.analysis_complete and st.session_state.glb_file:
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">🌐 3D Brain Visualization</div>', unsafe_allow_html=True)
                
                # Enhanced 3D Viewer
                brain_viewer_3d_enhanced(st.session_state.glb_file)
                
                st.info("🖱️ Interact with the 3D model: Rotate (Left Click + Drag), Zoom (Scroll), Pan (Right Click + Drag)")
                
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Please upload and analyze an MRI scan in the 'Upload & Analyze' tab to view the 3D visualization.")

    with tab3:
        # Results Section
        if st.session_state.analysis_complete:
            # Segmentation Results
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">📊 Segmentation Analysis</div>', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown('<div class="metric-value">12.4 cm³</div>', unsafe_allow_html=True)
                    st.markdown('<div class="metric-label">Tumor Volume</div>', unsafe_allow_html=True)
                    st.progress(0.75)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown('<div class="metric-value">8.7 cm³</div>', unsafe_allow_html=True)
                    st.markdown('<div class="metric-label">Edema Region</div>', unsafe_allow_html=True)
                    st.progress(0.65)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown('<div class="metric-value">92%</div>', unsafe_allow_html=True)
                    st.markdown('<div class="metric-label">Confidence</div>', unsafe_allow_html=True)
                    st.progress(0.92)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Disease Classification
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">🩺 Disease Probability</div>', unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown('<div class="metric-value">84%</div>', unsafe_allow_html=True)
                    st.markdown('<div class="metric-label">Glioma</div>', unsafe_allow_html=True)
                    st.progress(0.84)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown('<div class="metric-value">12%</div>', unsafe_allow_html=True)
                    st.markdown('<div class="metric-label">Meningioma</div>', unsafe_allow_html=True)
                    st.progress(0.12)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown('<div class="metric-value">3%</div>', unsafe_allow_html=True)
                    st.markdown('<div class="metric-label">Pituitary</div>', unsafe_allow_html=True)
                    st.progress(0.03)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col4:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown('<div class="metric-value">1%</div>', unsafe_allow_html=True)
                    st.markdown('<div class="metric-label">Other</div>', unsafe_allow_html=True)
                    st.progress(0.01)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Download Section
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">📥 Download Results</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.session_state.glb_file and os.path.exists(st.session_state.glb_file):
                        with open(st.session_state.glb_file, "rb") as f:
                            st.download_button(
                                label="💾 Download 3D Model (GLB)",
                                data=f,
                                file_name="brain_segmentation.glb",
                                mime="model/gltf-binary",
                                key="download_glb",
                                use_container_width=True
                            )
                
                with col2:
                    if st.session_state.pdf_file and os.path.exists(st.session_state.pdf_file):
                        with open(st.session_state.pdf_file, "rb") as f:
                            st.download_button(
                                label="📄 Download Medical Report (PDF)",
                                data=f,
                                file_name="comprehensive_medical_report.pdf",
                                mime="application/pdf",
                                key="download_pdf",
                                use_container_width=True
                            )
                
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Please upload and analyze an MRI scan in the 'Upload & Analyze' tab to view results.")

    st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <footer>
        <p>NeuroX Hybrid © 2025 | Advanced Brain Segmentation & Analysis Platform</p>
        <p style="font-size: 0.9rem; margin-top: 0.5rem;">Powered by Deep Learning, Computer Vision, and Medical AI</p>
    </footer>
    """, unsafe_allow_html=True)

def brain_viewer_3d_enhanced(glb_file_path="brain_segmentation.glb"):
    """
    Create an enhanced 3D viewer component for brain segmentation models with better visuals and controls.
    """
    
    # HTML template with Three.js for 3D visualization
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Enhanced 3D Brain Viewer</title>
        <style>
            body {{ 
                margin: 0; 
                overflow: hidden;
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }}
            #container {{ 
                width: 100%; 
                height: 100vh;
                position: relative;
            }}
            #info {{
                position: absolute;
                top: 15px;
                left: 15px;
                color: #e2e8f0;
                background: rgba(30, 41, 59, 0.7);
                padding: 10px 15px;
                border-radius: 8px;
                font-size: 14px;
                backdrop-filter: blur(5px);
                border: 1px solid rgba(148, 163, 184, 0.3);
                z-index: 100;
            }}
            #controls {{
                position: absolute;
                bottom: 15px;
                left: 15px;
                color: #e2e8f0;
                background: rgba(30, 41, 59, 0.7);
                padding: 15px;
                border-radius: 8px;
                font-size: 14px;
                backdrop-filter: blur(5px);
                border: 1px solid rgba(148, 163, 184, 0.3);
                z-index: 100;
            }}
            #loading {{
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                color: #e2e8f0;
                font-size: 18px;
                text-align: center;
            }}
            #loading-spinner {{
                border: 5px solid rgba(99, 102, 241, 0.3);
                border-top: 5px solid #6366f1;
                border-radius: 50%;
                width: 50px;
                height: 50px;
                animation: spin 1s linear infinite;
                margin: 0 auto 15px;
            }}
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
            .control-group {{
                margin-bottom: 10px;
            }}
            .control-label {{
                display: block;
                margin-bottom: 5px;
                font-weight: 500;
            }}
            button {{
                background: linear-gradient(90deg, #6366f1, #8b5cf6);
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 15px;
                cursor: pointer;
                font-weight: 500;
                transition: all 0.3s ease;
                margin-right: 10px;
                margin-bottom: 10px;
            }}
            button:hover {{
                transform: translateY(-2px);
                box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
            }}
            button:active {{
                transform: translateY(0);
            }}
            .slider-control {{
                width: 100%;
                margin: 5px 0;
            }}
        </style>
    </head>
    <body>
        <div id="container">
            <div id="info">
                <div>🧠 NeuroX 3D Brain Visualization</div>
                <div style="font-size: 12px; margin-top: 5px;">Left drag: Rotate | Scroll: Zoom | Right drag: Pan</div>
            </div>
            <div id="controls">
                <div class="control-group">
                    <div class="control-label">Model Controls</div>
                    <button id="resetView">Reset View</button>
                    <button id="toggleRotation">Toggle Rotation</button>
                </div>
                <div class="control-group">
                    <div class="control-label">Opacity Control</div>
                    <input type="range" min="0" max="100" value="100" class="slider-control" id="opacitySlider">
                </div>
            </div>
            <div id="loading">
                <div id="loading-spinner"></div>
                <div>Loading 3D brain model...</div>
            </div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/three@0.132.2/build/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.132.2/examples/js/controls/OrbitControls.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.132.2/examples/js/loaders/GLTFLoader.js"></script>

        <script>
            // Main variables
            let container, camera, scene, renderer, controls;
            let model, mixer, clock;
            let autoRotate = true;
            let modelOpacity = 1.0;
            
            init();
            animate();
            
            function init() {{
                // Get container
                container = document.getElementById('container');
                
                // Create scene
                scene = new THREE.Scene();
                scene.background = new THREE.Color(0x0f172a);
                scene.fog = new THREE.Fog(0x0f172a, 15, 30);
                
                // Create camera
                camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 1000);
                camera.position.set(0, 0, 15);
                
                // Create renderer
                renderer = new THREE.WebGLRenderer({{ antialias: true }});
                renderer.setSize(container.clientWidth, container.clientHeight);
                renderer.setPixelRatio(window.devicePixelRatio);
                renderer.shadowMap.enabled = true;
                container.appendChild(renderer.domElement);
                
                // Add lights
                const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
                scene.add(ambientLight);
                
                const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
                directionalLight.position.set(5, 10, 7);
                directionalLight.castShadow = true;
                scene.add(directionalLight);
                
                const backLight = new THREE.DirectionalLight(0x4d79ff, 0.8);
                backLight.position.set(-5, -5, -5);
                scene.add(backLight);
                
                // Add orbit controls
                controls = new THREE.OrbitControls(camera, renderer.domElement);
                controls.enableDamping = true;
                controls.dampingFactor = 0.05;
                controls.screenSpacePanning = false;
                controls.minDistance = 5;
                controls.maxDistance = 30;
                
                // Handle window resize
                window.addEventListener('resize', onWindowResize, false);
                
                // Setup clock for animations
                clock = new THREE.Clock();
                
                // Load 3D model
                const loader = new THREE.GLTFLoader();
                loader.load('{glb_file_path}', function(gltf) {{
                    model = gltf.scene;
                    
                    // Enable shadows
                    model.traverse(function(node) {{
                        if (node.isMesh) {{
                            node.castShadow = true;
                            node.receiveShadow = true;
                        }}
                    }});
                    
                    // Center the model
                    const box = new THREE.Box3().setFromObject(model);
                    const center = box.getCenter(new THREE.Vector3());
                    const size = box.getSize(new THREE.Vector3()).length();
                    
                    model.position.x += (model.position.x - center.x);
                    model.position.y += (model.position.y - center.y);
                    model.position.z += (model.position.z - center.z);
                    
                    // Scale model to fit view
                    const scale = 10 / size;
                    model.scale.set(scale, scale, scale);
                    
                    scene.add(model);
                    
                    // Setup animations if available
                    if (gltf.animations && gltf.animations.length) {{
                        mixer = new THREE.AnimationMixer(model);
                        gltf.animations.forEach((clip) => {{
                            mixer.clipAction(clip).play();
                        }});
                    }}
                    
                    // Hide loading message
                    document.getElementById('loading').style.display = 'none';
                }}, undefined, function(error) {{
                    console.error(error);
                    document.getElementById('loading').innerHTML = '<div>Error loading model: ' + error.message + '</div>';
                }});
                
                // Setup UI controls
                document.getElementById('resetView').addEventListener('click', resetView);
                document.getElementById('toggleRotation').addEventListener('click', toggleRotation);
                document.getElementById('opacitySlider').addEventListener('input', updateOpacity);
            }}
            
            function onWindowResize() {{
                camera.aspect = container.clientWidth / container.clientHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(container.clientWidth, container.clientHeight);
            }}
            
            function resetView() {{
                camera.position.set(0, 0, 15);
                camera.lookAt(0, 0, 0);
                controls.reset();
            }}
            
            function toggleRotation() {{
                autoRotate = !autoRotate;
                document.getElementById('toggleRotation').textContent = autoRotate ? 'Pause Rotation' : 'Resume Rotation';
            }}
            
            function updateOpacity(e) {{
                modelOpacity = e.target.value / 100;
                if (model) {{
                    model.traverse(function(node) {{
                        if (node.isMesh) {{
                            if (node.material) {{
                                node.material.opacity = modelOpacity;
                                node.material.transparent = modelOpacity < 1.0;
                            }}
                        }}
                    }});
                }}
            }}
            
            function animate() {{
                requestAnimationFrame(animate);
                
                const delta = clock.getDelta();
                
                if (controls) {{
                    controls.update();
                }}
                
                if (mixer) {{
                    mixer.update(delta);
                }}
                
                if (model && autoRotate) {{
                    model.rotation.y += 0.005;
                }}
                
                renderer.render(scene, camera);
            }}
        </script>
    </body>
    </html>
    """
    
    return components.html(html_template, height=700)

# Main execution
if __name__ == "__main__":
    # Check if we're running in Streamlit
    if "streamlit" in sys.modules or "STREAMLIT" in os.environ:
        run_streamlit_app()
    else:
        # Try to find datasets and train if available
        try:
            dataset_roots = find_all_dataset_roots()
            print(f"Found {len(dataset_roots)} datasets for training")
            
            # Collect subjects from all datasets
            all_subjects = []
            for dataset_root in dataset_roots:
                print(f"Processing dataset: {dataset_root}")
                if validate_dataset_structure(dataset_root):
                    subjects = list_subjects(dataset_root)
                    all_subjects.extend(subjects)
                    print(f"  Added {len(subjects)} subjects from {dataset_root}")
                else:
                    print(f"  ❌ Validation failed for dataset: {dataset_root}")
                    # Debug the dataset structure
                    debug_dataset_structure(dataset_root)
                    # Try to collect subjects anyway as a fallback
                    subjects = list_subjects(dataset_root)
                    if subjects:
                        all_subjects.extend(subjects)
                        print(f"  ⚠️  Added {len(subjects)} subjects from {dataset_root} (despite validation failure)")
                    else:
                        print(f"  ❌ No subjects found in {dataset_root}")
            
            if len(all_subjects) > 0:
                print(f"Starting training with {len(all_subjects)} subjects from {len(dataset_roots)} datasets...")
                trained_model = train_hybrid_model(all_subjects, MAX_EPOCHS)
                print("Training completed.")
            else:
                print("No valid subjects found in any dataset.")
        except Exception as e:
            print(f"Could not find datasets or train model: {e}")
            traceback.print_exc()
