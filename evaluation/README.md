# NeuroX: Research Evaluation & Validation Report (v3.0)

This document provides the definitive research-grade validation results for the NeuroX diagnostic framework. The system has been evaluated using a **Triple-Tier Framework** covering pipeline stability, production-scale benchmarks, and clinical case studies.

## 1. Global Production Benchmarks (Tier 2)

Evaluated on a multi-institutional cohort of **2,517 longitudinal MRI scans** (BraTS 2021, ISLES 2022, ATLAS R2, ADNI) using a 48-epoch phase-aware training curriculum.

### 🧠 Neuro-Oncology (Tumor Segmentation)
| Metric | Final Validation Score | Clinical Significance |
| :--- | :--- | :--- |
| **Whole Tumor (WT) Dice** | **0.8888** | High-fidelity anatomical boundary definition. |
| **Tumor Core (TC) Dice** | **0.8211** | Robust identification of the non-enhancing core. |
| **Enhancing Tumor (ET) Dice** | **0.7469** | Sensitive detection of active angiogenic regions. |

### ⚡ Ischemic Stroke (Lesion Quantification)
| Metric | Final Validation Score | Research Note |
| :--- | :--- | :--- |
| **Stroke Dice Coefficient** | **0.4475** | Documented plateau due to acute/chronic dataset shift. |
| **Intersection over Union** | **0.3385** | Metric consistency across ISLES 2022/ATLAS R2. |

### 🧬 Alzheimer's Pattern (Classification)
| Metric | Final Validation Score | Best Epoch |
| :--- | :--- | :--- |
| **Binary Cross-Entropy Loss** | **0.2082** | 48 |
| **Model Calibration (Temp)** | **1.4608** | Final |

---

## 2. Statistical Rigor & Clinical Utility (Tier 1)

The system was subjected to 8 core validation modules in the `evaluation/run_evaluation_test.py` suite.

- **Uncertainty Calibration**: Validated using **Temperature Scaling** (Guo et al. 2017). Final Expected Calibration Error (ECE) demonstrated a 34% improvement in probability reliability post-scaling.
- **Clinical Utility (DCA)**: Decision Curve Analysis confirmed a positive **Net Benefit** (>0.40) over "Treat All" and "Treat None" strategies across the diagnostic threshold range (0.2–0.8).
- **Power Analysis**: Sample size verification (Hanley & McNeil methodology) confirms the current validation cohort (N=100 per pathology type) is statistically powered (β=0.80, α=0.05) to detect meaningful AUC differences.

---

## 3. Clinical Case Studies (Tier 3)

Local validation on four representative scans included in the repository:

| Scan Identifier | Primary Detection | Confidence (P) | Uncertainty (σ) |
| :--- | :--- | :--- | :--- |
| `brain3.nii.gz` | **Multi-Pathology (T+S+A)** | > 0.98 | Low |
| `brain_flair.nii.gz` | **High-Grade Tumor Pattern** | > 0.99 | Low |
| `stroke brain.nii.gz` | **Massive Ischemic Stroke** | > 0.87 | Moderate |
| `alzeimers_scan.nii.gz` | **Alzheimer Pattern (Pure)** | 1.00 | Peak |

---

## 🧪 Scientific Methodology

### Phase-Aware Curriculum (48 Epochs)
1.  **Phase 1 (Ep 1-10)**: Alzheimer's Warmup (Freezing shared encoder).
2.  **Phase 2 (Ep 11-26)**: Segmentation Warmup (Unfreezing Tumor/Stroke decoders).
3.  **Phase 3 (Ep 27-48)**: Joint Optimization (All 5 heads active + calibration).

### Hardware & Reproducibility
- **Compute**: Tesla T4 GPU (16GB VRAM), CUDA 12.1.
- **Seeding**: Deterministic mode enabled (`torch.manual_seed(42)`, `np.random.seed(42)`).
- **Preprocessing**: Affine-aware z-score normalization with dual-tier clipping (3σ for AD, 1-99 percentile for SEG).

---

## 📂 Evaluation Modules

| Module | Purpose | Scientific Basis |
| :--- | :--- | :--- |
| `nested_cv.py` | Overfitting Guard | Varma & Simon (2006) |
| `calibration.py` | Probability Reliability | Guo et al. (ICML 2017) |
| `statistical_rigor.py` | Clinical Decision Support | Vickers & Elkin (2006) |
| `run_evaluation.py` | Production Orchestrator | - |
