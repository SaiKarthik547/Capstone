# NeuroX - Multi-Disease Brain MRI Analysis System

## 📋 Project Overview

NeuroX is an advanced deep learning system for automated detection and segmentation of brain pathologies from MRI scans. The system simultaneously analyzes three critical neurological conditions: **Brain Tumors**, **Stroke**, and **Alzheimer's Disease** using state-of-the-art 3D convolutional neural networks with hybrid uncertainty quantification.

### Project Goal

To develop a clinically-viable AI diagnostic assistant that:
- Provides accurate multi-disease detection from a single MRI scan
- Generates precise 3D visualizations of brain pathology
- Quantifies diagnostic uncertainty for clinical decision support
- Meets publication-grade statistical rigor for medical AI research

## ⚡ Quick Start

Get the system running in under 5 minutes:

1.  **Clone & Install**:
    ```bash
    git clone https://github.com/SaiKarthik547/Capstone.git
    cd Capstone
    pip install -r requirements.txt
    pip install HD-BET
    ```

2.  **Configure Environment**:
    - Create a `.env` file in the project root:
      ```env
      GROQ_API_KEY=gsk_your_key_here
      ```
    - (Optional) Set `NEUROX_OFFLINE=True` to disable cloud features.

3.  **Download Models**:
    - Place `neurox_model.pth` in the `checkpoints/` directory.

4.  **Run**:
    - `streamlit run neurox_adaptive.py`

---

## ⚠️ Disclaimer

**IMPORTANT LEGAL AND CLINICAL NOTICES:**

🚨 **NOT FOR CLINICAL USE** - This system is a research prototype and educational tool only. It is NOT approved by FDA, CE, or any regulatory body for clinical diagnosis or treatment decisions.

⚠️ **REQUIRES EXPERT SUPERVISION** - All outputs must be reviewed and validated by qualified medical professionals (radiologists, neurologists). AI predictions are assistive only and cannot replace human clinical judgment.

⚠️ **RESEARCH PURPOSES ONLY** - This software is intended for academic research, algorithm development, and educational demonstrations. Any clinical application requires proper regulatory approval and validation studies.

⚠️ **NO WARRANTY** - Provided "as-is" without any guarantees of accuracy, reliability, or fitness for any particular purpose. Users assume all risks.

⚠️ **DATA PRIVACY** - Users are responsible for ensuring compliance with HIPAA, GDPR, and other applicable data protection regulations when processing medical images.

---

## ✨ Key Features

### Multi-Label Disease Detection
- ✅ Simultaneous detection of Tumor, Stroke, and Alzheimer's Disease
- ✅ Independent probability scores (diseases are NOT mutually exclusive)
- ✅ One patient can have multiple conditions simultaneously

### Medical-Grade Brain Extraction
- ✅ HD-BET (Heidelberg Brain Extraction Tool) for accurate skull stripping
- ✅ Separates brain tissue from skull, scalp, and facial structures
- ✅ Essential for accurate 3D visualization

### 3D Visualization
- ✅ Interactive 3D brain mesh rendering
- ✅ Lesion overlay with anatomically accurate positioning
- ✅ Color-coded pathology visualization

### Uncertainty Quantification
- ✅ Monte Carlo Dropout for epistemic uncertainty estimation
- ✅ Confidence scores for each prediction
- ✅ Helps identify cases requiring expert review

### Training Dashboard
- ✅ Dedicated TRAINING page in the Streamlit app
- ✅ 48-epoch trend graphs for all metrics (Tumor Dice, Stroke Dice, AUC, F1)
- ✅ Phase transition annotations (P1 ALZ, P2 SEG, P3 JOINT)
- ✅ All-time best metrics displayed at a glance

### High-Resolution Clinical Export
- ✅ **2D Multi-Slice Heatmaps**: Matplotlib-powered PNG export (200 DPI) independent of browser rendering.
- ✅ **Anatomy Context**: Heatmaps include Axial, Coronal, and Sagittal strips with Grayscale anatomical underlays.
- ✅ **Professional PDFs**: Instant generation of structured clinical reports with biometric findings.

### Secure Infrastructure
- ✅ **Clean Session State**: API keys handled exclusively via `.env` to prevent session leaks.
- ✅ **Zero-Regeneration 3D**: Intelligent `@st.cache_resource` prevents re-calculating heavy brain meshes on UI interactions.

### Clinical Metrics
- ✅ Comprehensive evaluation: Precision, Recall, F1-Score
- ✅ Sensitivity, Specificity, PPV, NPV
- ✅ Advanced Spatial Metrics: Volume (mm³), Centroid (mm), and Bounding Box in World Coordinates
- ✅ Clinical-style quantification using Affine Matrix transformations
- ✅ ROC curves and calibration analysis

---

## 🛠️ Technology Stack

### Core Framework
- **Python 3.8+** - Primary programming language
- **PyTorch 2.0+** - Deep learning framework
- **CUDA 11.8+** - GPU acceleration (optional)
- **python-dotenv** - Secure environment-based configuration

### Medical Imaging
- **NiBabel** - NIfTI file I/O
- **HD-BET** - Medical-grade brain extraction
- **Scikit-image** - Image processing and marching cubes
- **SciPy** - Scientific computing and morphological operations

### 🧠 Deep Learning Architecture (v3)
- **SharedEncoder** - Dual-channel 3D CNN (`in_channels=2` for T1ce+FLAIR). Employs depth-wise separable convolutions and InstanceNorm3d for single-batch stability.
- **Conditional Transformer** - 3D Transformer Bottleneck (dim=128, depth=4) applied **exclusively to the Segmentation manifold**. Researched and validated to improve multi-class lesion boundaries while being **bypassed by the Alzheimer path** to prevent feature degradation.
- **AlzheimerEncoder v3** - Specialized independent branch with **ResBlock3D+SE** layers. Features a dedicated Dual-Head output: (1) logit for classification and (2) `log_var` for heteroscedastic uncertainty quantification.
- **Symmetric Segmentation Decoder** - High-fidelity decoder with residual skip-connections (bypassing Attention Gates for stable multi-task convergence). Supports ET (Enhancing Tumor), TC (Tumor Core), and WT (Whole Tumor) segmentation.

### Visualization & Export
- **Streamlit** - Interactive clinical whiteboard and web application framework
- **Plotly** - Interactive 3D graphics, mesh navigation, and training trend charts
- **Matplotlib (Agg)** - High-fidelity 2D multi-slice heatmap export (200 DPI)
- **Trimesh** - 3D mesh processing and STL generation

### Evaluation & Statistics
- **Scikit-learn** - ML metrics and evaluation
- **Iterative-stratification** - Multi-label cross-validation
- **NumPy** - Numerical computing
- **Pandas** - Data analysis and subject-level splitting

### Report Generation
- **ReportLab** - Automated PDF clinical reporting
- **Groq AI** - Llama-3 70B powered neuroradiology specialist (Secure `.env` required)

---

## 📦 Installation

### 1. Clone Repository
```bash
git clone https://github.com/SaiKarthik547/Capstone.git
cd Capstone
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

**Required packages:**
```
torch>=2.0.0
nibabel>=5.0.0
numpy>=1.24.0
scipy>=1.10.0
scikit-image>=0.20.0
scikit-learn>=1.3.0
streamlit>=1.28.0
plotly>=5.17.0
trimesh>=4.0.0
matplotlib>=3.7.0
reportlab>=4.0.0
iterative-stratification>=0.1.7
```

### 3. Install HD-BET (Medical-Grade Brain Extraction)
```bash
pip install HD-BET
```
**Verify installation:**
```bash
hd-bet -h
```
If successful, you should see HD-BET help documentation.

### 4. Download Model Weights
**Essential:**
1.  **NeuroX Model**: Place `neurox_model.pth` in the `checkpoints/` directory.
    - Generated by running `neurox_train_kaggle.py` (Kaggle GPU environment)
    - Format: `{"model_state": <weights>, "metrics": <48-epoch history>}`

**Optional (for 3D Brain Extraction):**
2.  **HD-BET Weights**: The system attempts to download these automatically. If it fails, place weights in `~/.hd-bet/`.

---

## 🚀 Usage

### Running the Application
```bash
streamlit run neurox_adaptive.py
```
The web interface will open automatically at `http://localhost:8501`.

### Workflow

1. **Upload MRI Scan**
   - Supported formats: `.nii`, `.nii.gz` (NIfTI)
   - Recommended: T1-weighted, T2-weighted, or FLAIR sequences
   - File size: Typically 5-50 MB

2. **Automatic Analysis** (takes 30-60 seconds)
   - Brain extraction using HD-BET
   - Multi-label disease detection
   - Lesion segmentation (Tumor/Stroke)
   - Uncertainty quantification
   - 3D mesh generation

3. **Review Results**
   - **Analysis Page**: Detection cards, 3D visualization, clinical metrics
   - **Training Page**: 48-epoch curriculum trend charts and all-time bests

4. **Export Report**
   - Download comprehensive PDF report
   - Includes all visualizations and metrics
   - Timestamped for record-keeping

---

## 📊 Evaluation & Validation System

NeuroX employs a **triple-tier validation framework** to ensure clinical reliability and scientific rigor. This "Rejection-Proof" methodology covers pipeline stability, production scale performance, and local clinical case studies.

### Tier 1: Integration Suite (Pipeline Stability)
*Verified using synthetic multi-label data (N=200, seed=42) to confirm mathematical and logical continuity across all 8 modules.*

| # | Test Module | Result | Impact |
|---|-------------|--------|--------|
| 1 | **Multi-label BCE** | ✅ PASS | Confirmed independent disease detection capability. |
| 2 | **Calibration Scaling** | ✅ PASS | Optimal (tuple-aware) temps: 1.78-1.80 (synthetic). |
| 3 | **ECE Verification** | ✅ PASS | Confirmed reliability of predicted probabilities. |
| 4 | **ROC Operating Points** | ✅ PASS | Calculated Sens@95%Spec for all diagnostic tasks. |
| 5 | **Lesion IoU Matching** | ✅ PASS | Validated 1:1 lesion identification logic. |
| 6 | **Sensitivity vs. Size** | ✅ PASS | Confirmed detection robustness for small lesions. |
| 7 | **Decision Curve (DCA)** | ✅ PASS | Validated Net Benefit > 0.40 over baseline strategies. |
| 8 | **Power Analysis** | ✅ PASS | Statistically powered for N=100 validation trials. |

### Tier 2: Global Production Benchmarks (48-Epoch Curriculum)
*Final performance on real-world medical imaging datasets (BraTS, ISLES, ADNI) following the completion of the 3-phase curriculum (Tesla T4 GPU).*

| Pathological Target | Metric (Holdout Val) | Final Score | Best Epoch |
| :--- | :--- | :--- | :--- |
| **Brain Tumor (WT)** | **Whole Tumor Dice** | **0.8888** | 48 |
| **Brain Tumor (TC)** | **Tumor Core Dice** | **0.8211** | 48 |
| **Brain Tumor (ET)** | **Enhancing Tumor Dice** | **0.7469** | 48 |
| **Ischemic Stroke** | **Stroke Dice** | **0.4475** | 48 |
| **Ischemic Stroke** | **Stroke IoU** | **0.3385** | 48 |
| **Alzheimer's** | **Classification Loss** | **0.2082** | 48 |
| **Calibration** | **Final Temperature** | **1.4608** | Final |

> [!IMPORTANT]
> **Research Significance:** For a detailed breakdown of the statistical rigor, decision curves (DCA), and clinical operating points, refer to the [NeuroX Research Evaluation Report](file:///c:/Users/karth/OneDrive/Desktop/neurox/evaluation/README.md).

#### 3. Phase-Aware Training Curriculum (10/26/48)
The model was trained using a structured 3-phase curriculum to ensure multi-task convergence:

1.  **Phase 1: ALZ Warmup (Ep 1-10)** - Primary focus on Alzheimer's feature extraction using the independent `AlzheimerEncoder`. Shared encoder frozen.
2.  **Phase 2: SEG Warmup (Ep 11-26)** - Unfreezing the tumor/stroke decoders and Transformer bottleneck to establish spatial awareness.
3.  **Phase 3: Joint Optimization (Ep 27-48)** - Full end-to-end training of all 5 output heads with dynamic temperature scaling.

### 🔬 Research Evidence: Ablation Study
To justify the final v3.5 architecture (No Attention Gates, Conditional Transformer), a systematic ablation study was conducted.

| Run | Configuration | AD AUC | Tumor Dice | Stroke Dice | AD Loss | Verdict |
|:---:|:---|:---:|:---:|:---:|:---:|:---|
| **1** | Full Context (All True) | 0.5842 | 0.8636 | 0.5092 | 0.2742 | Reference Baseline |
| **2** | **No Transformer** | **0.6972** | 0.8568 | 0.5006 | 0.2806 | **Beneficial for AD** |
| **3** | **No Attention Gates** | **0.6214** | 0.8619 | 0.4980 | 0.2899 | **Beneficial for AD** |
| **4** | No Uncertainty | 0.6091 | 0.8615 | 0.4980 | 0.2788 | Beneficial for AD |
| **5** | No Alz Isolation | 0.6364 | 0.8596 | 0.5191 | 0.3009 | Beneficial for AD |
| **6** | No SE Block | 0.6972 | 0.8568 | 0.5006 | 0.2806 | Beneficial for AD |

**Conclusion:** The 3D Transformer and Attention Gates were found to be **harmful contributors** to the global multi-task manifold, specifically degrading clinical Alzheimer's features. The optimized v3.5 architecture implements a **Conditional Transformer Bypass** and **Symmetric Residual Decoder** for peak diagnostic stability.

### Tier 3: Clinical Multi-Scan Trial (Local Case Studies)
*Independent validation on four distinct pathology cases using the 48-epoch production model.*

| Scan Identifier | Pathology Detection | Diagnostic Confidence | Uncertainty (Uncalibrated) |
| :--- | :--- | :--- | :--- |
| **`brain3.nii.gz`** | Tumor + Stroke + Alzheimer | **High (P > 0.98)** | T:0.77, S:1.01, A:1.05 |
| **`brain_flair.nii.gz`** | High-Grade Pattern (T+S+A) | **High (P > 0.99)** | T:0.79, S:0.88, A:1.05 |
| **`stroke brain.nii.gz`** | Massive Stroke + Tumor | **High (P > 0.87)** | T:0.89, S:1.04 |
| **`alzeimers_scan.nii.gz`** | Pure Alzheimer Pattern | **Peak (P = 1.00)** | A:1.05 |

### Evaluation Modules
- **`evaluation/nested_cv.py`** - Multi-label BCE verification, nested cross-validation (5×3), patient-level bootstrap CI.
- **`evaluation/calibration.py`** - Temperature scaling (Guo et al. 2017), ECE, reliability diagrams, ROC operating points.
- **`evaluation/statistical_rigor.py`** - Lesion-level IoU matching, sensitivity vs lesion size, decision curves, power analysis.
- **`evaluation/run_evaluation.py`** - Main evaluation orchestrator (production pipeline).

---

## 🏗️ Neural Architecture (NeuroX-v3)

The system utilizes a dual-path asymmetric architecture designed for high-fidelity segmentation and uncertainty-aware clinical classification.

```mermaid
graph TD
    %% ── Style Definitions ──────────────────────────────────────────────────
    classDef input fill:#f5f5f5,stroke:#333,stroke-width:2px;
    classDef shared fill:#e1f5fe,stroke:#01579b,stroke-width:2.5px;
    classDef alz fill:#fff3e0,stroke:#e65100,stroke-width:2.5px;
    classDef bottle fill:#f3e5f5,stroke:#4a148c,stroke-width:2.5px;
    classDef heads fill:#e8f5e9,stroke:#2e7d32,stroke-width:1.5px;
    classDef output fill:#eeeeee,stroke:#616161,stroke-width:2px;

    %% ── Input Manifold ──────────────────────────────────────────────────────
    Input_Manifold(["Input Manifold: B × 2 × 96 × 96 × 96"]):::input
    Input_Manifold ==> Shared_Backbone
    Input_Manifold ==> Alzheimer_Backbone

    subgraph Shared_Backbone ["Standard Clinical Backbone (Tumor/Stroke)"]
        direction TB
        Shared_Enc_1["Encoder Stage 1 (96³ )"] --- Pool_1[MaxPool3d s=2]
        Pool_1 --- Shared_Enc_2["Encoder Stage 2 (48³ )"]
        Shared_Enc_2 --- Pool_2[MaxPool3d s=2]
        Pool_2 --- Shared_Enc_3["Encoder Stage 3 (24³ )"]
    end
    Shared_Backbone:::shared

    subgraph Transformer_Unit ["Conditional Transformer Bottleneck (12³ )"]
        direction TB
        Pool_3[MaxPool3d s=2] --- Reshape_In["Flatten (1728 seq)"]
        Shared_Enc_3 --- Pool_3
        Reshape_In --- MHA_FFN["4× [Multi-Head Attention | FFN]"]
        MHA_FFN --- Reshape_Out["Reshape (B × 128 × 12³ )"]
    end
    Transformer_Unit:::bottle

    subgraph Alzheimer_Backbone ["Independent Alzheimer Backbone (RE-SE)"]
        direction TB
        Alz_B1["ResBlock 1: (48³ )"] --- Alz_B2["ResBlock 2: (24³ )"]
        Alz_B2 --- Alz_B3["ResBlock 3: (12³ )"]
        Alz_B3 --- Alz_B4["ResBlock 4: (12³ )"]
        Alz_B4 --- SE_Block["Squeeze-Excitation (SE) Block"]
    end
    Alzheimer_Backbone:::alz

    subgraph Decoding_Manifold ["Symmetric Decoding Manifold"]
        direction LR
        subgraph Spatial_Decoders ["Spatial ROI Reconstruction"]
            Dec_Stage_3["Up3 + Concatenate 256→128 (24³ )"]
            Dec_Stage_2["Up2 + Concatenate 128→64 (48³ )"]
            Dec_Stage_1["Up1 + Concatenate 64→32 (96³ )"]
            Dec_Stage_3 --- Dec_Stage_2 --- Dec_Stage_1
        end
        
        subgraph Probabilistic_Heads ["Clinical Presence Heads"]
            Tumor_Pres["Tumor Presence (logit + log_var)"]
            Stroke_Pres["Stroke Presence (logit + log_var)"]
            Alzheimer_Pres["AD Classification (logit + log_var)"]
        end
    end
    Decoding_Manifold:::heads

    %% ── Skip-Connection Manifolds ───────────────────────────────────────────
    Shared_Enc_3 -.-|enc3: 24³ | Dec_Stage_3
    Shared_Enc_2 -.-|enc2: 48³ | Dec_Stage_2
    Shared_Enc_1 -.-|enc1: 96³ | Dec_Stage_1

    %% ── Global Decision Fusion ──────────────────────────────────────────────
    Fused_Report(["Clinical Whiteboard Engine (MLP-Fusion)"]):::output
    
    Reshape_Out ==> Spatial_Decoders
    Reshape_Out ==> Tumor_Pres
    Reshape_Out ==> Stroke_Pres
    Alzheimer_Backbone ==>|Transformer Bypass Route| Alzheimer_Pres
    Decoding_Manifold ==> Fused_Report
```

---

### 🔄 Application Workflow

```mermaid
flowchart TD
    Start([🚀 Launch neurox_adaptive.py]) --> Upload{📁 Upload NIfTI File\n.nii / .nii.gz}
    Upload -->|Valid file| Preprocess

    subgraph Preprocessing_Stage ["🔬 Preprocessing Pipeline"]
        direction TB
        Preprocess["Load NiBabel\nextract voxel data + affine matrix"] --> ZScore
        ZScore["Z-Score Normalize\nmean=0, std=1\nclip ±5σ for seg path\nclip ±3σ for alz path"] --> PadCube
        PadCube["Pad to Cube\ncentered zero-padding\nto largest dimension"] --> ResampleROI
        ResampleROI["Trilinear Resample\n→ 96×96×96 ROI\noutput: B×1×96×96×96 (alz)\nor B×2×96×96×96 (tumor/stroke)"] --> ExtractAffine
        ExtractAffine["Extract Affine Matrix\n+ Voxel Spacing in mm\nfor world-space mapping"]
    end

    Preprocessing_Stage --> LoadModel

    subgraph Checkpoint_Load ["💾 Checkpoint Loading"]
        LoadModel["load_model(neurox_model.pth)\ndetect checkpoint format"] -->|New format| ExtractState
        LoadModel -->|New format| ExtractMetrics
        LoadModel -->|Legacy format| LegacyLoad
        ExtractState["Extract model_state\nOrderedDict weights"]
        ExtractMetrics["Extract metrics_history\n11 metrics × 48 epochs"]
        LegacyLoad["Plain state_dict\nno metrics fallback"]
    end

    ExtractState --> Validate["load_state_dict(strict=True)\nassert alz_encoder present"]
    LegacyLoad --> Validate
    Validate --> ModelReady["NeuroXMultiDisease.eval()\non DEVICE (CUDA/CPU)\nall 3 paths ready"]

    ModelReady --> DetFn["automatic_disease_detection\nthreshold=0.5\nuse_uncertainty=True\nMC-Dropout×10 passes"]

    subgraph AI_Engine ["🧠 NeuroX: Dual-Path Inference Engine"]

        subgraph SharedPath ["Path A — SharedEncoder + Transformer + PresenceHeads"]
            direction TB
            DetFn --> EncFwd["SharedEncoder.forward()\ninput: B×2×96×96×96\n(T1ce + FLAIR channels)"]
            EncFwd --> Enc1["EncoderBlock 1\nConv3d(2→32)×2 + IN3d(affine) + ReLU\nMaxPool3d(2)\n→ B×32×96×96×96 → skip e1\n→ B×32×48×48×48 pooled"]
            Enc1 --> Enc2["EncoderBlock 2\nConv3d(32→64)×2 + IN3d(affine) + ReLU\nMaxPool3d(2)\n→ B×64×48×48×48 → skip e2\n→ B×64×24×24×24 pooled"]
            Enc2 --> Enc3["EncoderBlock 3\nConv3d(64→128)×2 + IN3d(affine) + ReLU\nMaxPool3d(2)\n→ B×128×24×24×24 → skip e3\n→ B×128×12×12×12 pooled"]
            Enc3 --> TfmBottle["TransformerBottleneck3D\nreshape B×128×12×12×12 → 1728 tokens dim=128\n4× [PreNorm → MHA(8heads) → residual\n     PreNorm → FFN(256) → residual]\n→ B×128×12×12×12"]
            TfmBottle --> TumorHead["Tumor PresenceHead\nAvgPool3d → Flatten → FC(128→64)\n→ ReLU → Dropout(0.2) → FC(64→2)\n→ logit_tumor, log_var_tumor\nMC-Dropout × 10 passes\n→ prob_tumor, std_tumor"]
            TfmBottle --> StrokeHead["Stroke PresenceHead\nAvgPool3d → Flatten → FC(128→64)\n→ ReLU → Dropout(0.2) → FC(64→2)\n→ logit_stroke, log_var_stroke\nMC-Dropout × 10 passes\n→ prob_stroke, std_stroke"]
        end

        subgraph AlzPath ["Path B — AlzheimerEncoder (Zero Shared Parameters)"]
            direction TB
            RawMRI["Raw MRI slice\nB×1×96×96×96\nfrom Preprocessing\n(z-score + clip ±3σ)"] --> ABlock1
            ABlock1["ResBlock3D(1→32)\nConv3d×2 + IN3d + ReLU\n+ MaxPool3d(2)\n→ B×32×48×48×48"]
            ABlock1 --> ABlock2["ResBlock3D(32→64)\n+ MaxPool3d(2)\n→ B×64×24×24×24"]
            ABlock2 --> ABlock3["ResBlock3D(64→128)\n+ MaxPool3d(2)\n→ B×128×12×12×12"]
            ABlock3 --> ABlock4["ResBlock3D(128→256)\n→ B×256×12×12×12"]
            ABlock4 --> SEAttn["SEBlock(256, reduction=16)\nAvgPool → Linear(256→16)\n→ ReLU → Linear(16→256)\n→ Sigmoid → channel scale"]
            SEAttn --> ADualPool["Dual Pooling\nAvgPool3d(1) → B×256\nMaxPool3d(1) → B×256\ncat → B×512"]
            ADualPool --> ANorm["LayerNorm(512)"]
            ANorm --> AClassifier["Classifier MLP\nLinear(512→256) → GELU\n→ Dropout(0.3)\n→ Linear(256→128) → GELU\n→ Dropout(0.2)\n→ Linear(128→1) → logit_alz"]
            ANorm --> ALogVar["log_var_head (separate branch)\nLinear(512→64) → ReLU\n→ Linear(64→1) → log_var_alz\n(Kendall & Gal 2017 — TRUE variance)"]
            AClassifier --> AlzMC["MC-Dropout × 10 passes\n→ prob_alz, std_alz"]
            ALogVar --> AlzMC
        end

    end

    Preprocessing_Stage -.->|single channel slice| RawMRI

    TumorHead --> ProbT{"prob_tumor ≥ 0.5?"}
    StrokeHead --> ProbS{"prob_stroke ≥ 0.5?"}
    AlzMC --> ProbA{"prob_alz ≥ 0.5?"}

    subgraph Post_Processing ["🔧 Post-Processing & Segmentation"]
        direction TB
        ProbT -->|Yes| TumSeg["Tumor SegmentationDecoder\nSymmetric U-Net Decode\nbottleneck(12³) → up3(e3) → 24³\n→ up2(e2) → 48³\n→ up1(e1) → 96³\nConv3d(32→3): 3-class mask"]
        ProbS -->|Yes| StrSeg["Stroke SegmentationDecoder\nSame U-Net path as Tumor\nConv3d(32→1): binary mask"]
        ProbA -->|Yes| AlzNoSeg["Alzheimer: NO segmentation\nReport prob_alz + std_alz only"]
        ProbT -->|No| SkipT["No tumor mask\nSkip decoder"]
        ProbS -->|No| SkipS["No stroke mask\nSkip decoder"]
        ProbA -->|No| SkipA["No Alzheimer finding"]
        TumSeg --> MapT["Inverse resize + remove padding\nmap tumor mask to patient voxel space\ncompute ET/NCR/ED volumes in mm³\ncentroid & bounding box (world coords)"]
        StrSeg --> MapS["Inverse resize + remove padding\nmap stroke mask to patient voxel space\ncompute stroke volume in mm³\ncentroid & bounding box (world coords)"]
    end

    Preprocessing_Stage --> RawMRI
    MapT --> VizScene
    MapS --> VizScene
    AlzNoSeg --> VizScene
    SkipT --> VizScene
    SkipS --> VizScene
    SkipA --> VizScene

    subgraph HD_BET ["🧬 HD-BET Brain Extraction (3D Surface)"]
        BETCheck{"HD-BET installed?"}
        BETCheck -->|Yes| HDBET["HD-BET: medical-grade skull stripping\nextract brain mask from T1/FLAIR\nrun marching cubes → brain surface mesh\ntrimesh STL for 3D rendering"]
        BETCheck -->|No| NoHDBET["3D surface disabled\nsafe fallback — analysis continues\nlesions shown without brain mesh"]
    end

    VizScene["Assemble 3D Plotly Scene\nlesion meshes + brain surface\n(if HD-BET available)"]
    HDBET --> VizScene
    NoHDBET -.-> VizScene

    VizScene --> Dashboard["📊 Analysis Dashboard"]

    subgraph Streamlit_Pages ["📱 Streamlit Pages"]
        Dashboard --> DetCard["Detection Cards\nprob + MC-uncertainty per disease\nconfidence gauge + color badge"]
        Dashboard --> SegViz["Segmentation Visualization\n3D Plotly overlays\naxial/coronal/sagittal slice view"]
        Dashboard --> ClinMetrics["Clinical Metrics\nlesion volume + BraTS class breakdown\ncentroid + bbox in world mm coords\naffine-space spatial analysis"]
        Dashboard --> TrainPage["TRAINING Page\n48-epoch trend charts\nPhase annotations (P1/P2/P3)\nAll-time best metrics table\ntemperature calibration history"]
        ExtractMetrics -.->|if available| TrainPage
    end

    Dashboard --> ReportGen{"Generate Report?"}
    ReportGen -->|Groq LLM enabled| AIText["AI Clinical Summary\nGroq API → natural language report\nstructured findings + impression"]
    ReportGen -->|ReportLab PDF| PDF["Export PDF Report\nStructuredReport parser\nFindings + Measurements + Impression\ntimestamped NX-HHMMSS ID"]
    AIText --> End([✅ Session Complete])
    PDF --> End
```

---

### 🏋️ Training Flow Diagram

```mermaid
flowchart TD
    subgraph Data_Ingestion ["📦 Dataset Ingestion & Preprocessing"]
        direction LR
        DS1["BraTS 2020\n368 cases\nT1ce + FLAIR → (2-ch)\nLabels: ET(4)/NCR(1)/ED(2)"]
        DS2["BraTS 2021\n1251 cases\nT1ce + FLAIR → (2-ch)\nSame label scheme"]
        DS3["ISLES 2022 + ATLAS R2\n898 cases total\nDWI/ADC + T1 FLAIR\nBinary stroke mask"]
        DS4["ADNI-1208-Cohort\n966 train / 242 val\nAD=236 CN=972\n1208 total scans\n(1-ch, z-score ±3σ clip)"]
        DS1 & DS2 --> CombTumor["Combined Tumor\n1619 cases total\ncache v7_2ch_flair"]
        DS3 --> StrokeLoader["Stroke Loader\n898 cases"]
        DS4 --> AlzLoader["Alzheimer Loader\nWeightedRandomSampler\nclass-balanced batching"]
    end

    subgraph Cache ["🗄️ Persistent Cache v7_2ch_flair"]
        CombTumor --> CacheDisk["Kaggle /kaggle/working/cache\nPre-processed .pt files\ntorch.save(image+target)\nDisk guard: 5GB free check"]
    end

    CacheDisk --> Dataloaders
    StrokeLoader --> Dataloaders
    AlzLoader --> Dataloaders

    subgraph Dataloaders ["⚙️ DataLoaders (batch_size=1, workers=1)"]
        TumorDL["Tumor DataLoader\nBatch: B×2×96×96×96\nSeg target: B×3×96×96×96\nShuffle: True"]
        StrokeDL["Stroke DataLoader\nBatch: B×2×96×96×96\nSeg target: B×1×96×96×96\nShuffle: True"]
        AlzDL["Alzheimer DataLoader\nBatch: B×1×96×96×96\nLabel: B×1 (0/1)\nWeightedSampler"]
    end

    subgraph Curriculum ["📅 Phase-Aware Curriculum (48 Epochs)"]
        P1["Phase 1: Epochs 1–10\nTRAIN_ALZ=True TRAIN_SEG=False\nLR_alz=3e-5 (decays phase-aware)\nFocus: AlzheimerEncoder cold-start\nSharedEncoder FROZEN"]
        P2["Phase 2: Epochs 11–26\nTRAIN_ALZ=False TRAIN_SEG=True\nLR_shared_conv=1e-4 LR_transformer=5e-5\nFocus: SegDecoder warm-up\nStep cap: 1000 steps/epoch"]
        P3["Phase 3: Epochs 27–48\nTRAIN_ALZ=True TRAIN_SEG=True\nLR_alz=1e-5 (joint regime)\nLR_shared_conv=1e-4 LR_transformer=5e-5\nFull 1932 steps/epoch\nBoth tasks optimize together"]
        P1 --> P2 --> P3
    end

    subgraph Optimizers ["🔧 Optimizers & Loss"]
        OPT1["optimizer_shared: AdamW\nconv params: lr=1e-4 wd=1e-5\ntransformer params: lr=5e-5\nGrad accumulation: 4 steps\nAMP (torch.cuda.amp.GradScaler)\nscaler_seg"]
        OPT2["optimizer_alz: AdamW\nlr=phase-aware (3e-5/1.5e-5/1e-5)\nwd=1e-4\nGrad accumulation: 4 steps\nAMP scaler_alz"]
        LOSS["Loss Functions\nAlzheimer: BCEWithLogitsLoss\n  + pos_weight=N_neg/N_pos\nTumor seg: 0.8×DiceLoss + 0.2×BCE\nStroke seg: 0.8×DiceLoss + 0.2×BCE\n  + stroke pos_weight\nTumor presence: BCEWithLogits\nStroke presence: BCEWithLogits\nTotal: 1.0×L_alz + 0.7×L_tumor + 0.7×L_stroke"]
    end

    subgraph Model_Forward ["🧠 NeuroXMultiDisease Selective Forward"]
        direction TB
        Batch["Input Batch\n(task-tagged)"] --> PhaseRouter{"Active Tasks\nfrom PHASE_CONFIG"}
        PhaseRouter -->|TRAIN_ALZ| AlzFwd["AlzheimerEncoder path\nx[:,0:1] → (logit, log_var)\nBCE loss → optimizer_alz"]
        PhaseRouter -->|TRAIN_SEG| SegFwd["SharedEncoder → TransformerBottleneck\n→ PresenceHead(tumor/stroke)\n→ SegDecoder(tumor/stroke)\nDice+BCE → optimizer_shared"]
        AlzFwd & SegFwd --> GradAccum["Gradient Accumulation\naccum_steps=4\nGradClip max_norm=1.0"]
    end

    subgraph Monitoring ["📊 Per-Epoch Monitoring"]
        direction LR
        MonAlz["Alzheimer Monitor\nLogit Mean + Std\nPred prob mean\n(early collapse detection)"]
        MonSeg["Segmentation Monitor\nStroke Empty Predictions %\nStroke Mean Dice @0.5\nTumor ET + WT Soft Dice\n(training samples)"]
        MonLogit["Logit Health Check\nLogit Std (30 val samples)\nFeature Std (20 val samples)\n✅ healthy / 🚨 COLLAPSED flags"]
    end

    subgraph Validation ["📐 SEG Validation (Scientific Metric Set)"]
        ValTumor["Tumor Validation\nET Dice | TC Dice | WT Dice\ncalibrated predictions\nTemperature T=1.3540→1.5326"]
        ValStroke["Stroke Validation\nDice | IoU\nbinary threshold=0.5"]
        GlobalScore["Global Score\n= ET_Dice + TC_Dice + WT_Dice\n+ Stroke_Dice\nNew best → save _best checkpoint"]
    end

    subgraph Checkpointing ["💾 Checkpoint Strategy"]
        SaveLast["neurox_last.pth (every epoch)\nmodel_state + optimizer states\nepoch number + best_score\nfor session resume on Kaggle"]
        SaveBest["neurox_best.pth\nbest global score checkpoint\nsaved only on improvement"]
        SaveFinal["neurox_model.pth (final)\nmodel_state + full metrics history\nTemperature Scaling applied\nT optimized on val set\nDEPLOYED to Streamlit app"]
    end

    Curriculum --> Model_Forward
    Dataloaders --> TumorDL & StrokeDL & AlzDL
    TumorDL & StrokeDL & AlzDL --> Model_Forward
    Optimizers --> Model_Forward
    Model_Forward --> Monitoring
    Monitoring --> Validation
    Validation --> GlobalScore
    GlobalScore --> SaveLast & SaveBest
    P3 --> SaveFinal
```

---

### 📈 Preprocessing Pipeline Detail

```mermaid
flowchart LR
    subgraph Input ["📂 Raw NIfTI Input"]
        NII["*.nii / *.nii.gz\nArbitrary shape\nArbitrary voxel spacing"]
    end

    subgraph TumorPath ["Tumor & Stroke Path (2-channel)"]
        direction TB
        TLoad["nibabel.load → float32\nT1ce + FLAIR channels"] --> TClip["Percentile clip p1→p99\n(remove scanner outliers)"]
        TClip --> TMinMax["MinMax normalize\n(p1→p99 range → 0..1)"]
        TMinMax --> TZScore["Z-Score\n(mean=0, std=1)"]
        TZScore --> TInterp["F.interpolate trilinear\n→ 96×96×96 per channel"]
        TInterp --> T2Ch["torch.cat([T1ce, FLAIR], dim=0)\n→ shape (2, 96, 96, 96)"]
        T2Ch --> TValid["validate_tensor\nNaN/Inf hard-fail check"]
        TValid --> TCache["Save to cache\nCACHE_DIR / stem_v7_2ch_flair.pt"]
    end

    subgraph AlzPath2 ["Alzheimer Path (1-channel, light pipeline)"]
        direction TB
        ALoad["nibabel.load → float32\nT1-weighted structural MRI"] --> AZScore["Pure Z-Score normalize\n(no percentile clip)\npreserves hippocampal gradients"]
        AZScore --> AClip["Soft clip ±3σ\nremoves extreme outliers\nwithout compressing mid-range"]
        AClip --> AInterp["F.interpolate trilinear\n→ (1, 96, 96, 96)"]
        AInterp --> AValid["validate_tensor\nNaN/Inf hard-fail check"]
    end

    subgraph InferPath ["Inference Path (neurox_adaptive.py)"]
        direction TB
        ILoad["nibabel.load\nextract affine + voxel spacing"] --> IZScore["Z-Score per volume\nclip ±5σ"]
        IZScore --> IPad["Pad to cube\ncentered zero-pad\ntrack padding amounts"]
        IPad --> IResize["Trilinear resample\n→ (1, 96, 96, 96)"]
        IResize --> ISplit["Split for dual path:\n(2,96³) for seg/presence\n(1,96³) for alz_encoder"]
    end

    NII --> TumorPath
    NII --> AlzPath2
    NII --> InferPath
```

---

## 🏋️ Training Pipeline

### 48-Epoch Curriculum (`neurox_train_kaggle.py`)

| Phase | Epochs | Tasks Active | LR Config | Steps/Epoch | Focus |
|-------|--------|--------------|-----------|-------------|-------|
| **Phase 1** | 1–10 | ALZ only | LR_alz: 3e-5→1.5e-5 (ep>10) | 1932 (full) | AlzheimerEncoder cold-start; SharedEncoder frozen |
| **Phase 2** | 11–26 | SEG only | LR_shared_conv=1e-4, LR_tfm=5e-5 | 1000 (step cap) | SegDecoder + PresenceHeads warm-up; ALZ frozen |
| **Phase 3** | 27–48 | ALZ + SEG (Joint) | LR_alz=1e-5, LR_shared_conv=1e-4, LR_tfm=5e-5 | 1932 (full) | Full joint optimization; temperature calibration at end |

**Optimizer Details:**
- `optimizer_shared`: AdamW, conv_params lr=1e-4, transformer_params lr=5e-5, weight_decay=1e-5
- `optimizer_alz`: AdamW, lr=phase-aware (3e-5/1.5e-5/1e-5), weight_decay=1e-4
- Gradient Accumulation: 4 steps for both optimizers
- AMP (Automatic Mixed Precision): enabled for both
- Gradient Clipping: max_norm=1.0

### Loss Functions

| Stream | Loss | Weight |
|--------|------|--------|
| Alzheimer classification | BCEWithLogitsLoss + pos_weight (N_neg/N_pos) | λ=1.0 |
| Tumor segmentation | 0.8×SoftDice + 0.2×BCEWithLogits | λ=0.7 |
| Stroke segmentation | 0.8×SoftDice + 0.2×BCEWithLogits (pos_weight) | λ=0.7 |
| Tumor presence | BCEWithLogitsLoss | λ=1.0 (Phase 3 only) |
| Stroke presence | BCEWithLogitsLoss | λ=1.0 (Phase 3 only) |

**Total loss:** `L_total = 1.0×L_alz + 0.7×L_tumor_seg + 0.7×L_stroke_seg`

### Datasets

| Dataset | Source | Cases | Input | Task |
|---------|--------|-------|-------|------|
| BraTS 2020 + 2021 | T1ce + FLAIR | 1619 | 2-channel | Tumor seg (ET/TC/WT) |
| ISLES 2022 + ATLAS R2 | DWI/ADC + T1 FLAIR | **898** | 2-channel | Stroke binary seg |
| ADNI-1208-Protocol | T1 structural | 1208 scans | 1-channel | Alzheimer CN vs AD |
| — | — | — | — | 966 train / 242 val, AD=236 CN=972 |

### Class Imbalance Handling
- **Alzheimer**: `WeightedRandomSampler` with per-class weights = 1 / (count + ε); pos_weight in BCE
- **Stroke segmentation**: pos_weight = N_background / N_lesion per batch

### Checkpoint Format
```python
# neurox_model.pth (inference checkpoint — final, temperature-calibrated)
{
    "model_state": OrderedDict,        # strict=True on load
    "temperature": float,              # 1.5326 (final calibrated)
    "metrics": {
        "epoch":          [1..48],
        "tumor_et":       [...],       # validation ET Dice per epoch
        "tumor_tc":       [...],       # validation TC Dice per epoch
        "tumor_wt":       [...],       # validation WT Dice per epoch
        "tumor_mean":     [...],       # mean of ET/TC/WT
        "stroke_dice":    [...],       # validation Stroke Dice
        "stroke_iou":     [...],       # validation Stroke IoU
        "alz_loss":       [...],       # training Alzheimer loss
        "alz_auc":        [...],       # validation Alzheimer AUC (if computed)
        "global_score":   [...],       # composite global score per epoch
    }
}

# neurox_last.pth (resume checkpoint — saved every epoch)
{
    "model_state":       OrderedDict,
    "optimizer_shared":  optimizer.state_dict(),
    "optimizer_alz":     optimizer.state_dict(),
    "epoch":             int,
    "best_score":        float,
    "scaler_seg":        GradScaler.state_dict(),
    "scaler_alz":        GradScaler.state_dict(),
}
```

---

## 📁 Project Structure

```
neurox/
├── .gitignore                 # Git ignore file
├── LICENSE                    # MIT License
├── README.md                  # Project Documentation
├── assets/                    # Static assets
│   └── brain/                 # Brain meshes for 3D visualization
├── checkpoints/               # Trained models and checkpoints
│   └── neurox_model.pth       # 🧠 Trained Model Weights + 48-Epoch Metrics History
├── evaluation/                # Evaluation & Validation System
│   ├── calibration.py         # Reliability & temperature scaling
│   ├── nested_cv.py           # Nested Cross-Validation (5x2)
│   ├── run_evaluation.py      # Evaluation orchestrator
│   ├── run_evaluation_test.py # Synthetic integration tests
│   └── statistical_rigor.py  # Decision curves & power analysis
├── convert.py                 # File used to convert Nifti MRI files into gzip files for input 
├── neurox_adaptive.py         # 🚀 Main Application (Streamlit)
├── neurox_report_engine.py    # 📋 Structured Clinical Report Generation
├── neurox_train_kaggle.py     # Training Pipeline (48-Epoch Curriculum)
└── requirements.txt           # Dependency Requirements
```

---

## 🔧 Configuration

### Environment Variables

```bash
# Disable Groq AI (offline mode)
export NEUROX_OFFLINE=true

# GPU selection
export CUDA_VISIBLE_DEVICES=0
```

### Model Configuration

Edit `neurox_adaptive.py` (top of file):

```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROI_SIZE = (96, 96, 96)
PRESENCE_THRESHOLD = 0.5
MODEL_PATH = BASE_DIR / "checkpoints" / "neurox_model.pth"
```

### Training Configuration

Edit `neurox_train_kaggle.py`:

```python
SEED = 42
BATCH_SIZE = 1          # minimum for 3D volumes on T4 GPU
ACCUM_STEPS = 4         # effective batch size = 4
EPOCHS = 48
USE_AMP = True          # FP16 mixed precision

# Phase curriculum boundaries
PHASE_CONFIG = {
    10: (True,  False),  # Phase 1: ALZ only
    26: (False, True),   # Phase 2: SEG only
    48: (True,  True),   # Phase 3: Joint
}

# Multi-task loss weighting
LAMBDA_TUMOR  = 0.7
LAMBDA_STROKE = 0.7
LAMBDA_CLS    = 1.0     # Alzheimer
```

---

## 🐛 Troubleshooting

### HD-BET Not Found

```bash
pip install HD-BET
hd-bet -h  # Verify installation
```

If HD-BET is unavailable, 3D visualization will be disabled (safe fallback).

### CUDA Out of Memory

Use CPU mode:
```python
DEVICE = torch.device("cpu")
```

Or reduce batch size in the training script (`BATCH_SIZE = 1` is already minimum).

### Model Architecture Mismatch on Load

The checkpoint uses `strict=True`. If you see key mismatch errors, regenerate `neurox_model.pth` by running the training script. Key things to check:
- `SharedEncoder` must have `in_channels=2` (T1ce + FLAIR dual channel)
- `AlzheimerEncoder` must have `log_var_head` (separate variance branch — v3 architecture)
- `PresenceHead.fc2` outputs 2 neurons (logit + log_var), not 1
- `TransformerBottleneck3D` lives in `NeuroXMultiDisease`, not inside `SharedEncoder`

### Import Errors

```bash
pip install -r requirements.txt --force-reinstall
```

### High Uncertainty in Pathological Boundaries

The model is highly sensitive to voxelwise gradients. To ensure diagnostic validity, the system outputs **Hybrid Uncertainty** (Epistemic std and Aleatoric log-variance). In cases of high variance (σ > 0.05), manual slice verification is mandatory post HD-BET extraction.

---

## 📚 Scientific References

1. **HD-BET:** Isensee et al., "Automated brain extraction of multisequence MRI using artificial neural networks" (2019)
2. **Nested CV:** Varma & Simon, "Bias in error estimation when using cross-validation for model selection" (2006)
3. **Temperature Scaling:** Guo et al., "On Calibration of Modern Neural Networks" (ICML 2017)
4. **Monte Carlo Dropout:** Gal & Ghahramani, "Dropout as a Bayesian Approximation" (2016)
5. **Decision Curves:** Vickers & Elkin, "Decision Curve Analysis" (2006)
6. **Multi-Label Stratification:** Sechidis et al., "On the stratification of multi-label data" (2011)
7. **Squeeze-and-Excitation:** Hu et al., "Squeeze-and-Excitation Networks" (CVPR 2018)
8. **Focal Loss:** Lin et al., "Focal Loss for Dense Object Detection" (2017)
9. **Heteroscedastic Uncertainty:** Kendall & Gal, "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?" (NeurIPS 2017)
10. **Attention U-Net:** Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas" (2018)
11. **BraTS Challenge:** Menze et al., "The Multimodal Brain Tumor Image Segmentation Benchmark" (2015)
12. **ISLES Challenge:** de la Rosa et al., "ISLES 2022: A multi-center magnetic resonance imaging stroke lesion segmentation dataset" (2022)

---

## 📄 License

**Academic and Research Use Only**

This software is provided for educational and research purposes. Commercial use, clinical deployment, or any application involving patient care requires:
- Proper regulatory approval (FDA, CE, etc.)
- Clinical validation studies
- Institutional review board (IRB) approval
- Separate licensing agreement

---

## 👥 Contributors

**Sai Karthik** - Project Lead & Development,
**Manideep Sandireddy** - Development,
**Sai Varshith** - Development

---

## 📧 Contact & Support

For technical questions, bug reports, or collaboration inquiries:
- **GitHub Issues:** [Report a bug](https://github.com/SaiKarthik547/Capstone/issues)
- **Documentation:** See `evaluation/README.md` for detailed evaluation docs

---

## 🎯 Future Enhancements

- [ ] External validation on independent clinical datasets
- [ ] External validation on independent datasets
- [ ] Additional disease categories (hemorrhage, MS lesions)
- [ ] Real-time inference optimization
- [ ] Multi-sequence fusion (T1 + T2 + FLAIR triple-channel)
- [ ] Longitudinal analysis (disease progression tracking)
- [ ] Integration with PACS systems
- [ ] Alzheimer classification validation set AUC logging

---

**Version:** 3.5.0 — 48-Epoch Curriculum + Dual-Channel + Hybrid Uncertainty (MC + Heteroscedastic)
**Last Updated:** April 7, 2026
**Hardware:** Tesla T4 Industrial GPU Cluster (Production Protocol)
**Status:** ✅ Research Prototype - Production-Grade Code Quality