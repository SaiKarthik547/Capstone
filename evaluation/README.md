# NeuroX Evaluation Modules

This directory contains the rejection-proof evaluation system for NeuroX.

## Modules

### `nested_cv.py`
- Multi-label BCE verification
- True nested cross-validation (5 outer × 3 inner folds)
- Multi-label stratification (Sechidis et al.)
- Patient-level bootstrap confidence intervals

### `calibration.py`
- Uncertainty-aware temperature scaling (logit/log-variance compatible)
- Brier score + Expected Calibration Error (ECE)
- Reliability diagrams (calibration curves)
- ROC operating points (Sens@95%Spec, Spec@95%Sens)
- Cost-ratio sensitivity analysis

### `statistical_rigor.py`
- Lesion-level IoU matching (component-wise detection)
- Sensitivity vs lesion size curves
- Decision Curve Analysis (clinical utility)
- Power analysis (Hanley & McNeil method)

### `run_evaluation.py`
- Main evaluation pipeline orchestrator
- Runs all phases sequentially
- Generates final evaluation report

## Installation

```bash
pip install iterative-stratification
```

## Usage

```python
from evaluation.run_evaluation import NeuroXEvaluationPipeline

# Initialize pipeline
pipeline = NeuroXEvaluationPipeline(output_dir="./evaluation_results")

# Run phases
# Phase 1: Verify nested multi-label presence keys
pipeline.run_phase1_verification(model, dataset)

# Phase 2: True nested CV (extracting logits from (logit, log_var) tuples)
results = pipeline.run_phase2_nested_cv(model_class, dataset, hyperparams, train_fn, eval_fn)

# Phase 3: Uncertainty-aware calibration
calib = pipeline.run_phase3_calibration(logits_dict, labels_dict)

# Phase 4: Clinical operating points
thresh = pipeline.run_phase4_thresholds(labels_dict, calib['calibrated_probs'])

# Phase 7: Decision Curve Analysis (Clinical Utility)
dca = pipeline.run_phase7_decision_curves(labels_dict, calib['calibrated_probs'])

# Generate rejection-proof final report
pipeline.generate_final_report()
```

## Output Structure

```
evaluation_results/
├── phase1_verification.json
├── phase3_calibration.json
├── phase4_thresholds.json
├── phase7_decision_curves.json
├── nested_cv/
│   └── nested_cv_results.json
├── calibration/
│   ├── tumor_reliability.png
│   ├── stroke_reliability.png
│   └── alzheimer_reliability.png
├── cost_sensitivity/
│   ├── tumor_cost_sensitivity.png
│   ├── stroke_cost_sensitivity.png
│   └── alzheimer_cost_sensitivity.png
├── decision_curves/
│   ├── tumor_decision_curve.png
│   ├── stroke_decision_curve.png
│   └── alzheimer_decision_curve.png
└── FINAL_EVALUATION_REPORT.md
```

## Scientific Basis

All methods are based on peer-reviewed literature:

- **Nested CV:** Varma & Simon (2006), Cawley & Talbot (2010)
- **Multi-label Stratification:** Sechidis et al. (2011)
- **Temperature Scaling:** Guo et al. (ICML 2017)
- **Decision Curves:** Vickers & Elkin (2006)
- **Power Analysis:** Hanley & McNeil (1982), Jung (2024)
- **Bootstrap CI:** Efron & Tibshirani (1993)

## Rejection-Proof Features

✅ **Statistical Reviewers:** Nested CV, patient-level bootstrap, power analysis  
✅ **Clinical Reviewers:** Decision curves, operating points, lesion-level metrics  
✅ **Methodological Reviewers:** Ablation studies, uncertainty calibration justification  
✅ **Reproducibility Reviewers:** Frozen external validation, explicit framing, deterministic mode
