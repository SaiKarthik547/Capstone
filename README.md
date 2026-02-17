# NeuroX - Multi-Disease Brain MRI Analysis System

## 📋 Project Overview

NeuroX is an advanced deep learning system for automated detection and segmentation of brain pathologies from MRI scans. The system simultaneously analyzes three critical neurological conditions: **Brain Tumors**, **Stroke**, and **Alzheimer's Disease** using state-of-the-art 3D convolutional neural networks with uncertainty quantification.

### Project Goal

To develop a clinically-viable AI diagnostic assistant that:
- Provides accurate multi-disease detection from a single MRI scan
- Generates precise 3D visualizations of brain pathology
- Quantifies diagnostic uncertainty for clinical decision support
- Meets publication-grade statistical rigor for medical AI research

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

### Clinical Metrics
- ✅ Comprehensive evaluation: Precision, Recall, F1-Score
- ✅ Sensitivity, Specificity, PPV, NPV
- ✅ Volumetric measurements in mm³ and mL
- ✅ ROC curves and calibration analysis

---

## 🛠️ Technology Stack

### Core Framework
- **Python 3.8+** - Primary programming language
- **PyTorch 2.0+** - Deep learning framework
- **CUDA 11.8+** - GPU acceleration (optional)

### Medical Imaging
- **NiBabel** - NIfTI file I/O
- **HD-BET** - Medical-grade brain extraction
- **Scikit-image** - Image processing and marching cubes
- **SciPy** - Scientific computing and morphological operations

### Deep Learning Architecture
- **3D CNN Encoder** - Shared feature extraction
- **Transformer Bottleneck** - Global context modeling
- **Multi-Head Architecture** - Independent disease classifiers
- **Attention-Gated U-Net** - Segmentation decoders
- **InstanceNorm3d** - Normalization for small-batch stability

### Visualization
- **Streamlit** - Web application framework
- **Plotly** - Interactive 3D graphics
- **Trimesh** - 3D mesh processing
- **Matplotlib** - 2D plotting and analysis

### Evaluation & Statistics
- **Scikit-learn** - ML metrics and evaluation
- **Iterative-stratification** - Multi-label cross-validation
- **NumPy** - Numerical computing
- **Pandas** - Data analysis (optional)

### Report Generation
- **ReportLab** - PDF report creation
- **Groq AI** - Optional AI-powered insights (requires API key)

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

The trained model file (`neurox_multihead_final.pth`, 21.6 MB) should be placed in the project root directory.

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
   - **Detection Tab**: Probability scores for each disease
   - **3D Visualization**: Interactive brain mesh with lesions
   - **Metrics Tab**: Volumetric measurements and statistics
   - **AI Insights**: Optional AI-generated clinical summary

4. **Export Report**
   - Download comprehensive PDF report
   - Includes all visualizations and metrics
   - Timestamped for record-keeping

---

## 📊 Evaluation System

The project includes a comprehensive evaluation framework for publication-grade validation:

### Running Tests

```bash
python run_evaluation_test.py
```

**Expected output:**
```
✅ TEST 1 PASSED: Comprehensive metrics
✅ TEST 2 PASSED: Temperature scaling  
✅ TEST 3 PASSED: ROC operating points
✅ TEST 4 PASSED: Decision curve analysis
✅ TEST 5 PASSED: Power analysis
```

### Evaluation Modules

- **`evaluation/nested_cv.py`** - Nested cross-validation, multi-label stratification
- **`evaluation/calibration.py`** - Temperature scaling, calibration metrics
- **`evaluation/statistical_rigor.py`** - IoU matching, decision curves, power analysis
- **`evaluation/run_evaluation.py`** - Main evaluation orchestrator

---


---

## 🏗️ System Architecture

### 🧠 NeuroX Model Architecture

The core of NeuroX is a multi-head 3D Convolutional Neural Network with a Transformer bottleneck, designed for simultaneous multi-disease detection and segmentation.

```mermaid
classDiagram
    direction TB
    
    class StreamlitApp {
        +run_streamlit_app()
        +render_ui()
    }
    
    class Preprocessing {
        +load_and_preprocess_nifti()
        +apply_hdbet_brain_extraction()
        +generate_brain_mask_otsu()
    }
    
    class NeuroXModel {
        +SharedEncoder (CNN + Transformer)
        +PresenceHeads (Tumor, Stroke, Alzheimer)
        +SegmentationDecoders (Tumor, Stroke)
        +forward(x)
        +detect_presence()
        +segment_lesions()
    }

    class SharedEncoder {
        +Conv3DBlocks
        +TransformerBottleneck3D
        +InstanceNorm3D
    }
    
    class InferencePipeline {
        +automatic_disease_detection()
        +perform_segmentation()
        +uncertainty_quantification(MC Dropout)
    }

    class Visualization {
        +create_3d_visualization()
        +create_slice_view()
        +map_segmentation_to_original_space()
    }
    
    StreamlitApp --> Preprocessing : 1. Load NIfTI
    StreamlitApp --> InferencePipeline : 2. Run Analysis
    InferencePipeline --> NeuroXModel : Uses
    NeuroXModel *-- SharedEncoder : Feature Extraction
    NeuroXModel *-- PresenceHeads : Classification
    NeuroXModel *-- SegmentationDecoders : Localisation
    StreamlitApp --> Visualization : 3. Render 3D/2D
```

### 🔄 Application Workflow

The following flowchart illustrates the complete user journey from MRI upload to report generation, highlighting the conditional logic for different disease types and processing steps.

```mermaid
flowchart TD
    Start([🚀 Start Application]) --> Upload{📂 Upload NIfTI}
    Upload -->|Valid File| Preprocess[⚙️ Preprocessing]
    
    subgraph Preprocessing_Stage [Image Processing]
        Preprocess --> Normalize[Z-Score Normalization]
        Normalize --> ROI[Extract 96³ ROI Tensor]
        Normalize --> Affine[Extract Affine Matrix & Spacing]
    end
    
    ROI --> Inference[🧠 NeuroX AI Inference]
    
    subgraph AI_Engine [NeuroX Multi-Disease Model]
        Inference --> Encoder[Shared 3D Encoder]
        Encoder --> Bottleneck[Transformer Bottleneck]
        
        Bottleneck --> Head1[Tumor Presence Head]
        Bottleneck --> Head2[Stroke Presence Head]
        Bottleneck --> Head3[Alzheimer Presence Head]
        
        Head1 -->|Context Vector| Seg1[Tumor Segmentation Decoder]
        Head2 -->|Context Vector| Seg2[Stroke Segmentation Decoder]
    end
    
    Head1 -->|Logits| Prob1{Tumor Detected?}
    Head2 -->|Logits| Prob2{Stroke Detected?}
    Head3 -->|Logits| Prob3{Alzheimer Detected?}
    
    subgraph Post_Processing [Analysis & Refinement]
        Prob1 -->|Yes| Mask1[Generate Tumor Mask]
        Prob2 -->|Yes| Mask2[Generate Stroke Mask]
        Prob3 -->|Yes| NoSeg["No Segmentation (Alzheimer)"]
        
        Mask1 --> Clean1[Morphological Cleaning]
        Mask2 --> Clean2[Morphological Cleaning]
        
        Clean1 --> Map1[Map to Patient Space]
        Clean2 --> Map2[Map to Patient Space]
    end
    
    Preprocess --> SkullStrip{HD-BET Available?}
    SkullStrip -->|Yes| HDBET[Medical-Grade Extraction]
    SkullStrip -->|No| Fail[3D Visualization Disabled]
    
    HDBET --> BrainSurf[Generate Brain Surface Mesh]
    
    Map1 --> VizScene[Assemble 3D Scene]
    Map2 --> VizScene
    NoSeg --> VizScene
    BrainSurf --> VizScene
    
    VizScene --> Dashboard[📊 Clinical Dashboard]
    Fail -.-> Dashboard
    
    Dashboard --> ReportGen{📄 Generate Report}
    
    ReportGen -->|Groq AI| AI_Text[AI Clinical Summary]
    ReportGen -->|ReportLab| PDF[Export PDF Report]
    
    AI_Text --> End([✅ End Session])
    PDF --> End
```

---

## 📁 Project Structure

```
neurox/
├── neurox_adaptive.py              # Main Streamlit application
├── neurox_train_kaggle.py          # Training script
├── neurox_multihead_final.pth      # Trained model weights (21.6 MB)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── evaluation/                     # Evaluation system
│   ├── nested_cv.py               # Nested CV framework
│   ├── calibration.py             # Calibration & thresholds
│   ├── statistical_rigor.py       # Statistical analysis
│   ├── run_evaluation.py          # Main orchestrator
│   └── README.md                  # Evaluation docs
│
├── run_evaluation_test.py         # Test suite
│
└── assets/                        # Static assets
    └── brain/                     # Brain atlas files
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

Edit `neurox_adaptive.py` (lines 94-98):

```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROI_SIZE = (96, 96, 96)
PRESENCE_THRESHOLD = 0.5
MODEL_PATH = r"path/to/neurox_multihead_final.pth"
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

Or reduce batch size in training script.

### Import Errors

```bash
pip install -r requirements.txt --force-reinstall
```

---

## 📚 Scientific References

1. **HD-BET:** Isensee et al., "Automated brain extraction of multisequence MRI using artificial neural networks" (2019)
2. **Nested CV:** Varma & Simon, "Bias in error estimation when using cross-validation for model selection" (2006)
3. **Temperature Scaling:** Guo et al., "On Calibration of Modern Neural Networks" (ICML 2017)
4. **Monte Carlo Dropout:** Gal & Ghahramani, "Dropout as a Bayesian Approximation" (2016)
5. **Decision Curves:** Vickers & Elkin, "Decision Curve Analysis" (2006)
6. **Multi-Label Stratification:** Sechidis et al., "On the stratification of multi-label data" (2011)

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

**Sai Karthik** - Project Lead & Development

---

## 📧 Contact & Support

For technical questions, bug reports, or collaboration inquiries:
- **GitHub Issues:** [Report a bug](https://github.com/SaiKarthik547/Capstone/issues)
- **Documentation:** See `evaluation/README.md` for detailed evaluation docs

---

## 🎯 Future Enhancements

- [ ] External validation on independent datasets
- [ ] Additional disease categories (hemorrhage, MS lesions)
- [ ] Real-time inference optimization
- [ ] Multi-sequence fusion (T1 + T2 + FLAIR)
- [ ] Longitudinal analysis (disease progression tracking)
- [ ] Integration with PACS systems

---

**Version:** 1.5.0  
**Last Updated:** February 11, 2026  
**Status:** ✅ Research Prototype - Production-Grade Code Quality