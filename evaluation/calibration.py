"""
NeuroX Rejection-Proof Evaluation System
Phase 3-4: Calibration & Thresholds

Implements:
- Temperature scaling (justified vs matrix/Dirichlet)
- Brier score + ECE + reliability diagrams
- ROC operating points (Sens@95%Spec, Spec@95%Sens)
- Cost-ratio sensitivity analysis
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    brier_score_loss, roc_curve, confusion_matrix,
    precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3: TEMPERATURE SCALING CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════

class TemperatureScaling(nn.Module):
    """Temperature scaling for multi-label calibration.
    
    Justification:
    - Simple (1 param/disease) → low overfitting risk
    - Empirically robust for medical imaging (Guo et al. 2017)
    - Preferred over matrix/Dirichlet for small validation sets
    
    Scientific Basis:
    - Guo et al., "On Calibration of Modern Neural Networks" (ICML 2017)
    - Nixon et al., "Measuring Calibration in Deep Learning" (NeurIPS 2019)
    """
    
    def __init__(self, num_diseases=3):
        super().__init__()
        # Separate temperature per disease (minimal parameters)
        self.temperatures = nn.Parameter(torch.ones(num_diseases) * 1.5)
    
    def forward(self, logits, disease_idx):
        """Apply temperature scaling to logits for specific disease."""
        return logits / self.temperatures[disease_idx]
    
    def fit(
        self,
        logits_dict: Dict[str, torch.Tensor],
        labels_dict: Dict[str, torch.Tensor],
        disease_names: List[str],
        lr: float = 0.01,
        max_iter: int = 50
    ) -> Dict[str, float]:
        """Optimize temperatures on calibration subset.
        
        CRITICAL: Uses BCEWithLogitsLoss (not CrossEntropy) for multi-label.
        
        Args:
            logits_dict: Dict of logits per disease {disease: tensor}
            labels_dict: Dict of labels per disease {disease: tensor}
            disease_names: List of disease names
        
        Returns:
            Dict of optimal temperatures per disease
        """
        optimizer = torch.optim.LBFGS(self.parameters(), lr=lr, max_iter=max_iter)
        
        def eval_loss():
            optimizer.zero_grad()
            total_loss = 0
            
            for idx, disease in enumerate(disease_names):
                logits = logits_dict[disease]
                labels = labels_dict[disease].float()
                
                # Apply temperature scaling
                scaled_logits = self.forward(logits, idx)
                
                # ✅ CORRECT: BCEWithLogitsLoss for multi-label
                loss = F.binary_cross_entropy_with_logits(scaled_logits, labels)
                total_loss += loss
            
            total_loss.backward()
            return total_loss
        
        optimizer.step(eval_loss)
        
        # Return optimal temperatures
        return {
            disease: float(self.temperatures[idx])
            for idx, disease in enumerate(disease_names)
        }


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> float:
    """Compute Expected Calibration Error (ECE).
    
    ECE measures the difference between predicted probabilities and actual frequencies.
    
    Returns:
        ECE score (lower is better, 0 = perfect calibration)
    """
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            # Accuracy in bin
            accuracy_in_bin = y_true[in_bin].mean()
            # Average confidence in bin
            avg_confidence_in_bin = y_prob[in_bin].mean()
            # Add weighted difference to ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob_before: np.ndarray,
    y_prob_after: np.ndarray,
    disease_name: str,
    save_path: Path,
    n_bins: int = 10
):
    """Plot reliability diagram (calibration curve) before/after calibration.
    
    Shows: Predicted probability vs observed frequency
    Perfect calibration = diagonal line
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Before calibration
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    
    bin_accs_before = []
    bin_confs_before = []
    bin_counts_before = []
    
    for i in range(n_bins):
        in_bin = (y_prob_before > bin_boundaries[i]) & (y_prob_before <= bin_boundaries[i+1])
        if in_bin.sum() > 0:
            bin_accs_before.append(y_true[in_bin].mean())
            bin_confs_before.append(y_prob_before[in_bin].mean())
            bin_counts_before.append(in_bin.sum())
        else:
            bin_accs_before.append(0)
            bin_confs_before.append(bin_centers[i])
            bin_counts_before.append(0)
    
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax1.bar(bin_centers, bin_accs_before, width=1/n_bins, alpha=0.7, 
            label='Observed frequency', edgecolor='black')
    ax1.plot(bin_centers, bin_confs_before, 'ro-', label='Mean predicted probability')
    ax1.set_xlabel('Predicted Probability', fontsize=12)
    ax1.set_ylabel('Observed Frequency', fontsize=12)
    ax1.set_title(f'{disease_name} - Before Calibration', fontsize=14)
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # After calibration
    bin_accs_after = []
    bin_confs_after = []
    
    for i in range(n_bins):
        in_bin = (y_prob_after > bin_boundaries[i]) & (y_prob_after <= bin_boundaries[i+1])
        if in_bin.sum() > 0:
            bin_accs_after.append(y_true[in_bin].mean())
            bin_confs_after.append(y_prob_after[in_bin].mean())
        else:
            bin_accs_after.append(0)
            bin_confs_after.append(bin_centers[i])
    
    ax2.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax2.bar(bin_centers, bin_accs_after, width=1/n_bins, alpha=0.7,
            label='Observed frequency', edgecolor='black')
    ax2.plot(bin_centers, bin_confs_after, 'ro-', label='Mean predicted probability')
    ax2.set_xlabel('Predicted Probability', fontsize=12)
    ax2.set_ylabel('Observed Frequency', fontsize=12)
    ax2.set_title(f'{disease_name} - After Calibration', fontsize=14)
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Reliability diagram saved: {save_path}")


def report_calibration_metrics(
    y_true: np.ndarray,
    y_prob_before: np.ndarray,
    y_prob_after: np.ndarray,
    disease_name: str
) -> dict:
    """Report comprehensive calibration metrics.
    
    Returns:
        Dict with Brier score, ECE, and improvement metrics
    """
    # Brier score
    brier_before = brier_score_loss(y_true, y_prob_before)
    brier_after = brier_score_loss(y_prob_after)
    brier_improvement = brier_before - brier_after
    
    # ECE
    ece_before = compute_expected_calibration_error(y_true, y_prob_before)
    ece_after = compute_expected_calibration_error(y_true, y_prob_after)
    ece_improvement = ece_before - ece_after
    
    # Prevalence
    prevalence = y_true.mean()
    
    print(f"\n{disease_name.upper()} - Calibration Metrics:")
    print(f"  Prevalence: {prevalence:.3f}")
    print(f"  Brier Score:")
    print(f"    Before: {brier_before:.4f}")
    print(f"    After: {brier_after:.4f}")
    print(f"    Improvement: {brier_improvement:.4f} ({100*brier_improvement/brier_before:.1f}%)")
    print(f"  ECE:")
    print(f"    Before: {ece_before:.4f}")
    print(f"    After: {ece_after:.4f}")
    print(f"    Improvement: {ece_improvement:.4f}")
    
    # Interpretation
    if brier_improvement > 0.01:
        interpretation = "Substantial improvement"
    elif brier_improvement > 0.005:
        interpretation = "Moderate improvement"
    else:
        interpretation = "Minimal improvement"
    
    print(f"  Interpretation: {interpretation}")
    
    return {
        "brier_before": brier_before,
        "brier_after": brier_after,
        "brier_improvement": brier_improvement,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "ece_improvement": ece_improvement,
        "prevalence": prevalence,
        "interpretation": interpretation
    }


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 4: ROC OPERATING POINTS & THRESHOLD SELECTION
# ═══════════════════════════════════════════════════════════════════════════

def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float = 0.5,
    disease_name: str = ""
) -> dict:
    """Compute comprehensive classification metrics.
    
    Returns:
        Dict with sensitivity, specificity, precision, recall, F1, PPV, NPV, etc.
    """
    y_pred = (y_scores >= threshold).astype(int)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Sensitivity (Recall, True Positive Rate)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    recall = sensitivity
    
    # Specificity (True Negative Rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Precision (Positive Predictive Value)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    ppv = precision
    
    # Negative Predictive Value
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    
    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    
    # False Positive Rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    # False Negative Rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    return {
        "threshold": float(threshold),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "ppv": float(ppv),
        "npv": float(npv),
        "accuracy": float(accuracy),
        "fpr": float(fpr),
        "fnr": float(fnr)
    }


def compute_roc_operating_points(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    disease_name: str
) -> dict:
    """Compute ROC operating points (clinically defensible thresholds).
    
    Reports:
    - Sensitivity @ 95% Specificity
    - Specificity @ 95% Sensitivity
    - Comprehensive metrics at both operating points
    
    These are standard operating points in clinical AI (no arbitrary cost ratios).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # Sensitivity @ 95% Specificity
    target_spec = 0.95
    spec = 1 - fpr
    
    # Find threshold closest to 95% specificity
    idx_spec95 = np.argmin(np.abs(spec - target_spec))
    threshold_spec95 = thresholds[idx_spec95]
    sens_at_spec95 = tpr[idx_spec95]
    actual_spec95 = spec[idx_spec95]
    
    # Compute comprehensive metrics at this threshold
    metrics_spec95 = compute_comprehensive_metrics(
        y_true, y_scores, threshold_spec95, disease_name
    )
    
    # Specificity @ 95% Sensitivity
    target_sens = 0.95
    
    # Find threshold closest to 95% sensitivity
    idx_sens95 = np.argmin(np.abs(tpr - target_sens))
    threshold_sens95 = thresholds[idx_sens95]
    spec_at_sens95 = spec[idx_sens95]
    actual_sens95 = tpr[idx_sens95]
    
    # Compute comprehensive metrics at this threshold
    metrics_sens95 = compute_comprehensive_metrics(
        y_true, y_scores, threshold_sens95, disease_name
    )
    
    print(f"\n{disease_name.upper()} - ROC Operating Points:")
    print(f"  Sensitivity @ 95% Specificity:")
    print(f"    Threshold: {threshold_spec95:.3f}")
    print(f"    Sensitivity: {sens_at_spec95:.3f}")
    print(f"    Specificity: {actual_spec95:.3f}")
    print(f"    Precision: {metrics_spec95['precision']:.3f}")
    print(f"    F1-Score: {metrics_spec95['f1_score']:.3f}")
    print(f"  Specificity @ 95% Sensitivity:")
    print(f"    Threshold: {threshold_sens95:.3f}")
    print(f"    Specificity: {spec_at_sens95:.3f}")
    print(f"    Sensitivity: {actual_sens95:.3f}")
    print(f"    Precision: {metrics_sens95['precision']:.3f}")
    print(f"    F1-Score: {metrics_sens95['f1_score']:.3f}")
    
    return {
        "sens_at_spec95": {
            "threshold": float(threshold_spec95),
            "sensitivity": float(sens_at_spec95),
            "specificity": float(actual_spec95),
            "comprehensive_metrics": metrics_spec95
        },
        "spec_at_sens95": {
            "threshold": float(threshold_sens95),
            "specificity": float(spec_at_sens95),
            "sensitivity": float(actual_sens95),
            "comprehensive_metrics": metrics_sens95
        }
    }


def cost_ratio_sensitivity_analysis(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    disease_name: str,
    save_path: Path
) -> dict:
    """Perform cost-ratio sensitivity analysis.
    
    Shows how optimal threshold changes with different FN/FP cost assumptions.
    More defensible than fixed cost ratios.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # Test range of FN/FP cost ratios
    cost_ratios = [
        {"name": "Equal costs", "fp": 1.0, "fn": 1.0},
        {"name": "FN 2x worse", "fp": 1.0, "fn": 2.0},
        {"name": "FN 3x worse", "fp": 1.0, "fn": 3.0},
        {"name": "FN 5x worse", "fp": 1.0, "fn": 5.0},
        {"name": "FN 10x worse", "fp": 1.0, "fn": 10.0},
    ]
    
    results = []
    
    for ratio in cost_ratios:
        fn_rate = 1 - tpr
        total_cost = ratio["fp"] * fpr + ratio["fn"] * fn_rate
        optimal_idx = np.argmin(total_cost)
        
        results.append({
            "cost_ratio_name": ratio["name"],
            "fp_cost": ratio["fp"],
            "fn_cost": ratio["fn"],
            "optimal_threshold": float(thresholds[optimal_idx]),
            "sensitivity": float(tpr[optimal_idx]),
            "specificity": float(1 - fpr[optimal_idx])
        })
    
    # Plot sensitivity analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    fn_costs = [r["fn_cost"] for r in results]
    thresholds_opt = [r["optimal_threshold"] for r in results]
    sensitivities = [r["sensitivity"] for r in results]
    
    ax1.plot(fn_costs, thresholds_opt, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('FN/FP Cost Ratio', fontsize=12)
    ax1.set_ylabel('Optimal Threshold', fontsize=12)
    ax1.set_title(f'{disease_name} - Threshold vs Cost Ratio', fontsize=14)
    ax1.grid(alpha=0.3)
    
    ax2.plot(fn_costs, sensitivities, 'o-', linewidth=2, markersize=8, label='Sensitivity')
    ax2.set_xlabel('FN/FP Cost Ratio', fontsize=12)
    ax2.set_ylabel('Sensitivity at Optimal Threshold', fontsize=12)
    ax2.set_title(f'{disease_name} - Sensitivity vs Cost Ratio', fontsize=14)
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n{disease_name.upper()} - Cost Ratio Sensitivity Analysis:")
    for r in results:
        print(f"  {r['cost_ratio_name']}: threshold={r['optimal_threshold']:.3f}, "
              f"sens={r['sensitivity']:.3f}, spec={r['specificity']:.3f}")
    
    print(f"\n✅ Cost sensitivity plot saved: {save_path}")
    
    return {
        "sensitivity_analysis": results,
        "recommendation": "Use ROC operating points or clinical context for threshold selection"
    }


if __name__ == "__main__":
    print("NeuroX Rejection-Proof Evaluation System - Phase 3-4")
    print("="*80)
    print("\nThis module provides:")
    print("  1. Temperature scaling calibration")
    print("  2. Brier score + ECE + reliability diagrams")
    print("  3. ROC operating points (Sens@95%Spec, Spec@95%Sens)")
    print("  4. Cost-ratio sensitivity analysis")
    print("\nImport and use in your evaluation pipeline.")
