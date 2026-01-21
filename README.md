# Capstone


═══════════════════════════════════════════════════════════════════════════════
NeuroX Adaptive – Multi-Disease MRI Analysis System
Research & Educational Demonstration System
═══════════════════════════════════════════════════════════════════════════════

IMPORTANT ACADEMIC DISCLAIMERS:

1. PURPOSE
   This system performs PATTERN RECOGNITION, not clinical diagnosis.
   Outputs are for research, education, and algorithm demonstration only.

2. PRESENCE DETECTION ≠ DISEASE CONFIRMATION
   - "Tumor presence" = imaging characteristics consistent with abnormal tissue
   - "Stroke presence" = signal patterns suggestive of ischemic changes  
   - "Alzheimer presence" = global volumetric features indicating neurodegeneration
   
   ALL require expert clinical interpretation and histopathological confirmation.

3. ALZHEIMER ANALYSIS METHODOLOGY
   - Pattern-based presence detection using whole-brain features
   - NOT cortical thickness measurement
   - NOT voxel-wise atrophy segmentation
   - Rationale: ADNI dataset provides only subject-level labels (CN/MCI/AD)
   - No ground-truth voxel-level pathology exists for neurodegenerative disease

4. TRAINING DATASETS
   - Brain Tumor: BraTS 2020 (multimodal MRI, multinational)
   - Ischemic Stroke: ISLES 2022 (DWI/ADC sequences)
   - Alzheimer's: ADNI-derived (T1-weighted structural MRI)

5. TECHNICAL CONTRIBUTIONS
   - Multi-task learning with shared encoder
   - Conditional segmentation (disease-specific decoders)
   - Transformer-based volumetric encoding
   - Uncertainty-aware inference (Monte-Carlo Dropout)
   - Temperature-scaled confidence calibration

6. ARCHITECTURE DECISIONS
   InstanceNorm3d (not BatchNorm3d):
   - Training uses batch_size=2 (GPU memory constraint)
   - InstanceNorm provides stable normalization for small batches
   - Standard practice in medical imaging (nnU-Net, MONAI)
   
   Shared Encoder:
   - Single encoder for all three diseases
   - Improves multi-task generalization via shared representations
   - Reduces parameters while maintaining performance
   
   Transformer Bottleneck:
   - Captures long-range spatial dependencies
   - Handles variable lesion sizes via self-attention
   - Superior to pure CNN for 3D medical volumes

7. KNOWN LIMITATIONS (Honest Academic Assessment)
   - No skull stripping (model learns brain vs skull discrimination)
   - Fixed 96³ resolution (information loss for large native scans)
   - No multi-site harmonization (scanner-specific biases)
   - Single time-point analysis (no longitudinal tracking)
   - Imbalanced datasets: Tumor (300) >> Stroke (200) >> Alzheimer (100)
   - No external validation (same dataset train/val splits)

8. REGULATORY COMPLIANCE
   ⚠️  NOT FDA-approved
   ⚠️  NOT CE-marked
   ⚠️  NOT validated for clinical use
   
   For clinical deployment: IRB approval + prospective validation required.

9. INFERENCE vs TRAINING
   This file: Inference-only (no training logic)
   Training file: neurox_train_kaggle.py (separate, production-grade)
   
   Normalization consistency: InstanceNorm3d in both files
   Checkpoint compatibility: Loads with strict=False for partial matches

═══════════════════════════════════════════════════════════════════════════════

PRODUCTION IMPROVEMENTS (This Refactored Version):

✓ Normalization fix: InstanceNorm3d matches training pipeline
✓ Monte-Carlo Dropout: Epistemic uncertainty estimation
✓ Temperature scaling: Probability calibration
✓ Volumetric quantification: Clinical-style metrics
✓ Hard Alzheimer guards: No segmentation enforcement
✓ Deterministic mode: Reproducible results
✓ Offline mode: No internet dependencies
✓ Inference engine: Modular, auditable pipeline

═══════════════════════════════════════════════════════════════════════════════

For questions or academic collaboration:
Project: NeuroX - Final Year Medical Imaging AI
Institution: [Your University]
Supervisors: [Professor Names]

═══════════════════════════════════════════════════════════════════════════════
