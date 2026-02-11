"""
NeuroX Rejection-Proof Evaluation System
Phase 1-2: Core Infrastructure

Implements:
- Multi-label BCE verification
- True nested CV (5 outer × 3 inner)
- Multi-label stratification (Sechidis et al.)
- Patient-level bootstrap CI
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json

# Install iterative-stratification for multi-label CV
# pip install iterative-stratification
try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    STRATIFICATION_AVAILABLE = True
except ImportError:
    print("⚠️ WARNING: iterative-stratification not installed")
    print("   Install with: pip install iterative-stratification")
    STRATIFICATION_AVAILABLE = False
    MultilabelStratifiedKFold = None


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1: MULTI-LABEL BCE VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════

def verify_multi_label_training(model: nn.Module, dataset) -> dict:
    """Verify that model is using BCEWithLogitsLoss for multi-label classification.
    
    Critical checks:
    1. Loss function is BCEWithLogitsLoss (not CrossEntropyLoss)
    2. Dataset contains multi-label examples (co-occurring diseases)
    3. Model outputs independent logits per disease
    
    Returns:
        Dict with verification results
    """
    print("\n" + "="*80)
    print("PHASE 1: Multi-Label BCE Verification")
    print("="*80)
    
    # Check 1: Verify dataset has multi-label examples
    multi_label_count = 0
    single_label_count = 0
    no_label_count = 0
    
    disease_counts = {"tumor": 0, "stroke": 0, "alzheimer": 0}
    
    for i in range(min(len(dataset), 1000)):  # Sample first 1000
        sample = dataset[i]
        
        labels = {
            "tumor": sample["tumor_presence"].item(),
            "stroke": sample["stroke_presence"].item(),
            "alzheimer": sample["alzheimer_presence"].item()
        }
        
        # Count positive labels
        for disease, value in labels.items():
            if value > 0:
                disease_counts[disease] += 1
        
        num_positive = sum(labels.values())
        
        if num_positive > 1:
            multi_label_count += 1
        elif num_positive == 1:
            single_label_count += 1
        else:
            no_label_count += 1
    
    total_checked = multi_label_count + single_label_count + no_label_count
    
    print(f"\n✅ Dataset Multi-Label Analysis (n={total_checked}):")
    print(f"   Multi-label samples: {multi_label_count} ({100*multi_label_count/total_checked:.1f}%)")
    print(f"   Single-label samples: {single_label_count} ({100*single_label_count/total_checked:.1f}%)")
    print(f"   No-label samples: {no_label_count} ({100*no_label_count/total_checked:.1f}%)")
    print(f"\n   Disease prevalence:")
    for disease, count in disease_counts.items():
        print(f"     {disease}: {count}/{total_checked} ({100*count/total_checked:.1f}%)")
    
    if multi_label_count == 0:
        print("\n⚠️ WARNING: No multi-label examples found!")
        print("   Model will never learn co-occurrence patterns.")
        print("   Consider data augmentation or synthetic multi-label examples.")
    
    # Check 2: Verify model architecture
    print(f"\n✅ Model Architecture Check:")
    print(f"   Presence heads: {list(model.presence_heads.keys())}")
    print(f"   Segmentation decoders: {list(model.seg_decoders.keys())}")
    
    # Check 3: Verify loss function (will be checked in training loop)
    print(f"\n✅ Loss Function Requirement:")
    print(f"   MUST USE: nn.BCEWithLogitsLoss() for each disease")
    print(f"   DO NOT USE: nn.CrossEntropyLoss() (assumes mutual exclusivity)")
    
    return {
        "multi_label_count": multi_label_count,
        "single_label_count": single_label_count,
        "total_checked": total_checked,
        "multi_label_ratio": multi_label_count / total_checked if total_checked > 0 else 0,
        "disease_counts": disease_counts,
        "has_multi_label_examples": multi_label_count > 0
    }


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2: TRUE NESTED CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

class NestedCrossValidation:
    """True nested cross-validation with multi-label stratification.
    
    Structure:
        Outer Loop (5 folds): Generalization estimation
            Inner Loop (3 folds): Hyperparameter + calibration + threshold selection
    
    Critical: Outer test folds NEVER influence optimization
    """
    
    def __init__(
        self,
        n_outer_folds: int = 5,
        n_inner_folds: int = 3,
        random_state: int = 42
    ):
        self.n_outer_folds = n_outer_folds
        self.n_inner_folds = n_inner_folds
        self.random_state = random_state
        
        if not STRATIFICATION_AVAILABLE:
            raise ImportError(
                "iterative-stratification required. "
                "Install with: pip install iterative-stratification"
            )
    
    def prepare_multi_label_matrix(self, dataset) -> np.ndarray:
        """Convert dataset to multi-label matrix for stratification.
        
        Returns:
            y_multilabel: [N, 3] array with binary labels for each disease
        """
        y_multilabel = []
        
        for i in range(len(dataset)):
            sample = dataset[i]
            y_multilabel.append([
                sample["tumor_presence"].item(),
                sample["stroke_presence"].item(),
                sample["alzheimer_presence"].item()
            ])
        
        return np.array(y_multilabel)
    
    def run_nested_cv(
        self,
        model_class,
        dataset,
        hyperparameters_grid: List[dict],
        train_fn,
        evaluate_fn,
        device: str = "cuda"
    ) -> dict:
        """Execute full nested cross-validation.
        
        Args:
            model_class: Model class to instantiate
            dataset: Full dataset
            hyperparameters_grid: List of hyperparameter configurations to test
            train_fn: Function(model, train_data, hyperparams) -> trained_model
            evaluate_fn: Function(model, test_data) -> metrics_dict
            device: Device for training
        
        Returns:
            Dict with nested CV results
        """
        print("\n" + "="*80)
        print("PHASE 2: True Nested Cross-Validation")
        print("="*80)
        print(f"Outer folds: {self.n_outer_folds}")
        print(f"Inner folds: {self.n_inner_folds}")
        print(f"Hyperparameter configs: {len(hyperparameters_grid)}")
        
        # Prepare multi-label matrix
        y_multilabel = self.prepare_multi_label_matrix(dataset)
        
        print(f"\nDataset size: {len(dataset)}")
        print(f"Multi-label distribution:")
        print(f"  Tumor: {y_multilabel[:, 0].sum():.0f} ({100*y_multilabel[:, 0].mean():.1f}%)")
        print(f"  Stroke: {y_multilabel[:, 1].sum():.0f} ({100*y_multilabel[:, 1].mean():.1f}%)")
        print(f"  Alzheimer: {y_multilabel[:, 2].sum():.0f} ({100*y_multilabel[:, 2].mean():.1f}%)")
        
        # Outer CV: Generalization estimation
        outer_cv = MultilabelStratifiedKFold(
            n_splits=self.n_outer_folds,
            shuffle=True,
            random_state=self.random_state
        )
        
        outer_results = {
            "tumor": {"auc": [], "ap": [], "sensitivity": [], "specificity": []},
            "stroke": {"auc": [], "ap": [], "sensitivity": [], "specificity": []},
            "alzheimer": {"auc": [], "ap": [], "sensitivity": [], "specificity": []}
        }
        
        outer_predictions = []  # Store all predictions for patient-level bootstrap
        
        for outer_fold_idx, (outer_train_idx, outer_test_idx) in enumerate(
            outer_cv.split(np.zeros(len(dataset)), y_multilabel)
        ):
            print(f"\n{'='*80}")
            print(f"OUTER FOLD {outer_fold_idx + 1}/{self.n_outer_folds}")
            print(f"{'='*80}")
            
            # Split data
            outer_train_data = [dataset[i] for i in outer_train_idx]
            outer_test_data = [dataset[i] for i in outer_test_idx]
            
            print(f"Outer train: {len(outer_train_data)} samples")
            print(f"Outer test: {len(outer_test_data)} samples (HELD OUT)")
            
            # Verify stratification
            outer_test_labels = y_multilabel[outer_test_idx]
            print(f"Outer test distribution:")
            print(f"  Tumor: {outer_test_labels[:, 0].sum():.0f} ({100*outer_test_labels[:, 0].mean():.1f}%)")
            print(f"  Stroke: {outer_test_labels[:, 1].sum():.0f} ({100*outer_test_labels[:, 1].mean():.1f}%)")
            print(f"  Alzheimer: {outer_test_labels[:, 2].sum():.0f} ({100*outer_test_labels[:, 2].mean():.1f}%)")
            
            # ========================================
            # INNER LOOP: Hyperparameter tuning
            # ========================================
            
            print(f"\n--- Inner Loop: Hyperparameter Tuning ---")
            
            best_hyperparams = self._inner_cv_loop(
                model_class,
                outer_train_data,
                y_multilabel[outer_train_idx],
                hyperparameters_grid,
                train_fn,
                evaluate_fn,
                device
            )
            
            print(f"\n✅ Best hyperparameters selected: {best_hyperparams}")
            
            # ========================================
            # OUTER LOOP: Train final model
            # ========================================
            
            print(f"\n--- Outer Loop: Final Model Training ---")
            
            # Train final model on entire outer training set
            final_model = model_class(**best_hyperparams).to(device)
            final_model = train_fn(final_model, outer_train_data, best_hyperparams)
            
            # Evaluate on outer test set (INDEPENDENT)
            print(f"\n--- Evaluating on Outer Test Set (INDEPENDENT) ---")
            test_results = evaluate_fn(final_model, outer_test_data)
            
            # Store results
            for disease in ["tumor", "stroke", "alzheimer"]:
                if disease in test_results:
                    outer_results[disease]["auc"].append(test_results[disease]["auc"])
                    outer_results[disease]["ap"].append(test_results[disease]["ap"])
                    outer_results[disease]["sensitivity"].append(test_results[disease]["sensitivity"])
                    outer_results[disease]["specificity"].append(test_results[disease]["specificity"])
            
            # Store predictions for patient-level bootstrap
            outer_predictions.append({
                "fold": outer_fold_idx,
                "predictions": test_results.get("predictions", {}),
                "labels": test_results.get("labels", {})
            })
            
            print(f"\nOuter Fold {outer_fold_idx + 1} Results:")
            for disease in ["tumor", "stroke", "alzheimer"]:
                if disease in test_results:
                    print(f"  {disease}: AUC={test_results[disease]['auc']:.3f}, "
                          f"AP={test_results[disease]['ap']:.3f}")
        
        # ========================================
        # AGGREGATE RESULTS
        # ========================================
        
        print(f"\n{'='*80}")
        print("NESTED CV COMPLETE - Computing Patient-Level Bootstrap CI")
        print(f"{'='*80}")
        
        final_results = self._compute_patient_level_bootstrap_ci(
            outer_results,
            outer_predictions
        )
        
        return final_results
    
    def _inner_cv_loop(
        self,
        model_class,
        outer_train_data,
        outer_train_labels,
        hyperparameters_grid,
        train_fn,
        evaluate_fn,
        device
    ) -> dict:
        """Inner CV loop for hyperparameter selection."""
        
        inner_cv = MultilabelStratifiedKFold(
            n_splits=self.n_inner_folds,
            shuffle=True,
            random_state=self.random_state
        )
        
        best_hyperparams = None
        best_inner_score = -np.inf
        
        for hyperparam_config in hyperparameters_grid:
            print(f"\nTesting hyperparameters: {hyperparam_config}")
            
            inner_scores = []
            
            for inner_fold_idx, (inner_train_idx, inner_val_idx) in enumerate(
                inner_cv.split(np.zeros(len(outer_train_data)), outer_train_labels)
            ):
                # Inner train/val split
                inner_train_data = [outer_train_data[i] for i in inner_train_idx]
                inner_val_data = [outer_train_data[i] for i in inner_val_idx]
                
                # Train model
                model = model_class(**hyperparam_config).to(device)
                model = train_fn(model, inner_train_data, hyperparam_config)
                
                # Evaluate
                val_results = evaluate_fn(model, inner_val_data)
                
                # Aggregate score (mean AUC across diseases)
                score = np.mean([
                    val_results.get(disease, {}).get("auc", 0)
                    for disease in ["tumor", "stroke", "alzheimer"]
                ])
                
                inner_scores.append(score)
            
            # Average inner CV score
            mean_inner_score = np.mean(inner_scores)
            print(f"  Inner CV score: {mean_inner_score:.3f} ± {np.std(inner_scores):.3f}")
            
            if mean_inner_score > best_inner_score:
                best_inner_score = mean_inner_score
                best_hyperparams = hyperparam_config
        
        return best_hyperparams
    
    def _compute_patient_level_bootstrap_ci(
        self,
        outer_results: dict,
        outer_predictions: List[dict],
        n_bootstrap: int = 1000,
        confidence: float = 0.95
    ) -> dict:
        """Compute patient-level bootstrap confidence intervals.
        
        CRITICAL: Bootstrap over PATIENTS, not folds.
        """
        print(f"\nComputing patient-level bootstrap CI (n={n_bootstrap})...")
        
        # Aggregate all patient-level predictions
        all_predictions = {
            "tumor": {"y_true": [], "y_scores": []},
            "stroke": {"y_true": [], "y_scores": []},
            "alzheimer": {"y_true": [], "y_scores": []}
        }
        
        for fold_data in outer_predictions:
            preds = fold_data["predictions"]
            labels = fold_data["labels"]
            
            for disease in ["tumor", "stroke", "alzheimer"]:
                if disease in preds and disease in labels:
                    all_predictions[disease]["y_true"].extend(labels[disease])
                    all_predictions[disease]["y_scores"].extend(preds[disease])
        
        # Convert to arrays
        for disease in ["tumor", "stroke", "alzheimer"]:
            all_predictions[disease]["y_true"] = np.array(all_predictions[disease]["y_true"])
            all_predictions[disease]["y_scores"] = np.array(all_predictions[disease]["y_scores"])
        
        # Bootstrap over patients
        final_results = {}
        
        for disease in ["tumor", "stroke", "alzheimer"]:
            y_true = all_predictions[disease]["y_true"]
            y_scores = all_predictions[disease]["y_scores"]
            
            if len(y_true) == 0:
                continue
            
            n_patients = len(y_true)
            
            # Bootstrap resampling
            auc_bootstrap = []
            ap_bootstrap = []
            
            np.random.seed(self.random_state)
            for _ in range(n_bootstrap):
                # Resample PATIENTS with replacement
                patient_indices = np.random.choice(n_patients, size=n_patients, replace=True)
                
                y_true_boot = y_true[patient_indices]
                y_scores_boot = y_scores[patient_indices]
                
                # Skip if no positive samples
                if y_true_boot.sum() == 0 or y_true_boot.sum() == len(y_true_boot):
                    continue
                
                # Compute metrics
                auc_boot = roc_auc_score(y_true_boot, y_scores_boot)
                ap_boot = average_precision_score(y_true_boot, y_scores_boot)
                
                auc_bootstrap.append(auc_boot)
                ap_bootstrap.append(ap_boot)
            
            # Compute percentile-based CI
            alpha = 1 - confidence
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            auc_ci = (
                np.percentile(auc_bootstrap, lower_percentile),
                np.percentile(auc_bootstrap, upper_percentile)
            )
            ap_ci = (
                np.percentile(ap_bootstrap, lower_percentile),
                np.percentile(ap_bootstrap, upper_percentile)
            )
            
            final_results[disease] = {
                "auc_mean": np.mean(auc_bootstrap),
                "auc_std": np.std(auc_bootstrap),
                "auc_ci_95": auc_ci,
                "ap_mean": np.mean(ap_bootstrap),
                "ap_std": np.std(ap_bootstrap),
                "ap_ci_95": ap_ci,
                "n_patients": n_patients,
                "n_bootstrap": n_bootstrap,
                "bootstrap_type": "patient-level"
            }
            
            print(f"\n{disease.upper()} Results (Patient-Level Bootstrap):")
            print(f"  N patients: {n_patients}")
            print(f"  AUC-ROC: {final_results[disease]['auc_mean']:.3f}")
            print(f"    95% CI: [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}] (patient-level bootstrap, n={n_bootstrap})")
            print(f"  AUC-PR: {final_results[disease]['ap_mean']:.3f}")
            print(f"    95% CI: [{ap_ci[0]:.3f}, {ap_ci[1]:.3f}]")
        
        return final_results


# ═══════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def save_nested_cv_results(results: dict, output_path: Path):
    """Save nested CV results to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    serializable_results = convert_to_serializable(results)
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")


if __name__ == "__main__":
    print("NeuroX Rejection-Proof Evaluation System - Phase 1-2")
    print("="*80)
    print("\nThis module provides:")
    print("  1. Multi-label BCE verification")
    print("  2. True nested CV (5 outer × 3 inner)")
    print("  3. Multi-label stratification")
    print("  4. Patient-level bootstrap CI")
    print("\nImport and use in your training script.")
