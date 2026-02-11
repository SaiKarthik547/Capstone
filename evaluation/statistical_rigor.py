"""
NeuroX Rejection-Proof Evaluation System
Phase 5-7: Statistical Rigor

Implements:
- Lesion-level IoU matching (component-wise)
- Decision Curve Analysis (clinical utility)
- Power analysis (Hanley & McNeil)
- Volume-based thresholds (mm³)
"""

import numpy as np
import torch
from scipy.ndimage import label as scipy_label
from scipy import stats
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 6: LESION-LEVEL IOU MATCHING
# ═══════════════════════════════════════════════════════════════════════════

def compute_lesion_level_iou_matching(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    iou_threshold: float = 0.1,
    min_overlap_voxels: int = 10
) -> dict:
    """Lesion-level detection with IoU matching.
    
    CRITICAL:
    - Each GT lesion matched to at most one predicted lesion
    - Matching criterion: IoU ≥ threshold (default: 0.1)
    - Unmatched GT lesions = False Negatives
    - Unmatched predicted lesions = False Positives
    
    Scientific Basis:
    - BraTS challenge evaluation protocol
    - COCO detection metrics (object detection)
    - Medical image segmentation best practices
    
    Args:
        pred_mask: Predicted lesion mask (binary)
        gt_mask: Ground truth lesion mask (binary)
        iou_threshold: Minimum IoU for positive detection
        min_overlap_voxels: Minimum overlap voxels (noise filter)
    
    Returns:
        Dict with TP, FP, FN counts and matched lesions
    """
    # Separate into individual lesion components
    pred_labeled, n_pred = scipy_label(pred_mask)
    gt_labeled, n_gt = scipy_label(gt_mask)
    
    # Extract lesion properties
    pred_lesions = []
    for i in range(1, n_pred + 1):
        lesion_mask = (pred_labeled == i)
        centroid = np.array(np.where(lesion_mask)).mean(axis=1)
        pred_lesions.append({
            "id": i,
            "mask": lesion_mask,
            "centroid": centroid,
            "volume": lesion_mask.sum()
        })
    
    gt_lesions = []
    for i in range(1, n_gt + 1):
        lesion_mask = (gt_labeled == i)
        centroid = np.array(np.where(lesion_mask)).mean(axis=1)
        gt_lesions.append({
            "id": i,
            "mask": lesion_mask,
            "centroid": centroid,
            "volume": lesion_mask.sum()
        })
    
    # Compute IoU matrix (GT x Pred)
    iou_matrix = np.zeros((n_gt, n_pred))
    
    for i, gt_lesion in enumerate(gt_lesions):
        for j, pred_lesion in enumerate(pred_lesions):
            intersection = np.logical_and(gt_lesion["mask"], pred_lesion["mask"]).sum()
            union = np.logical_or(gt_lesion["mask"], pred_lesion["mask"]).sum()
            
            if union > 0:
                iou = intersection / union
            else:
                iou = 0
            
            iou_matrix[i, j] = iou
    
    # Greedy matching: Match GT lesions to predicted lesions
    matched_gt = set()
    matched_pred = set()
    matches = []
    
    # Sort by IoU (descending)
    iou_pairs = []
    for i in range(n_gt):
        for j in range(n_pred):
            if iou_matrix[i, j] >= iou_threshold:
                iou_pairs.append((i, j, iou_matrix[i, j]))
    
    iou_pairs.sort(key=lambda x: x[2], reverse=True)
    
    for gt_idx, pred_idx, iou in iou_pairs:
        if gt_idx not in matched_gt and pred_idx not in matched_pred:
            # Check minimum overlap
            intersection = np.logical_and(
                gt_lesions[gt_idx]["mask"],
                pred_lesions[pred_idx]["mask"]
            ).sum()
            
            if intersection >= min_overlap_voxels:
                matched_gt.add(gt_idx)
                matched_pred.add(pred_idx)
                matches.append({
                    "gt_id": gt_idx,
                    "pred_id": pred_idx,
                    "iou": iou,
                    "intersection_voxels": int(intersection)
                })
    
    # Compute TP, FP, FN
    tp = len(matches)
    fn = n_gt - len(matched_gt)
    fp = n_pred - len(matched_pred)
    
    # Sensitivity (lesion-level)
    sensitivity = tp / n_gt if n_gt > 0 else 0
    
    # Precision (lesion-level)
    precision = tp / n_pred if n_pred > 0 else 0
    
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "sensitivity": sensitivity,
        "precision": precision,
        "n_gt_lesions": n_gt,
        "n_pred_lesions": n_pred,
        "matches": matches,
        "iou_threshold": iou_threshold
    }


def sensitivity_vs_lesion_size_curve(
    predictions: List[dict],
    ground_truth: List[dict],
    voxel_spacing: Tuple[float, float, float],
    iou_threshold: float = 0.1
) -> dict:
    """Compute sensitivity vs lesion size with IoU matching.
    
    CORRECTED VERSION: Uses IoU matching, not simple overlap.
    
    Args:
        predictions: List of predicted masks
        ground_truth: List of GT masks
        voxel_spacing: Voxel spacing in mm (e.g., (0.95, 0.94, 1.20))
        iou_threshold: IoU threshold for matching
    
    Returns:
        Dict with sensitivity per size bin
    """
    voxel_volume_mm3 = np.prod(voxel_spacing)
    
    # Define size bins (mm³)
    size_bins = [0, 50, 100, 200, 500, 1000, 2000, 5000, np.inf]
    bin_labels = ["<50", "50-100", "100-200", "200-500", "500-1k", "1k-2k", "2k-5k", ">5k"]
    
    results = {
        label: {"tp": 0, "fn": 0, "sensitivity": 0, "total": 0}
        for label in bin_labels
    }
    
    for pred, gt in zip(predictions, ground_truth):
        # IoU matching
        matching_results = compute_lesion_level_iou_matching(
            pred["lesion_mask"],
            gt["lesion_mask"],
            iou_threshold=iou_threshold
        )
        
        # Separate GT lesions by size
        gt_labeled, n_gt = scipy_label(gt["lesion_mask"])
        
        for i in range(1, n_gt + 1):
            lesion_mask = (gt_labeled == i)
            lesion_voxels = lesion_mask.sum()
            lesion_volume_mm3 = lesion_voxels * voxel_volume_mm3
            
            # Find size bin
            bin_idx = np.digitize(lesion_volume_mm3, size_bins) - 1
            bin_label = bin_labels[bin_idx]
            
            # Check if this lesion was matched
            matched = any(m["gt_id"] == (i-1) for m in matching_results["matches"])
            
            if matched:
                results[bin_label]["tp"] += 1
            else:
                results[bin_label]["fn"] += 1
            
            results[bin_label]["total"] += 1
    
    # Compute sensitivity per bin
    for label in bin_labels:
        tp = results[label]["tp"]
        fn = results[label]["fn"]
        total = tp + fn
        
        if total > 0:
            results[label]["sensitivity"] = tp / total
        else:
            results[label]["sensitivity"] = np.nan
    
    print("\nSensitivity vs Lesion Size (IoU-based matching):")
    print(f"{'Size Range (mm³)':<20} {'Sensitivity':<15} {'N':<10}")
    print("-" * 45)
    for label in bin_labels:
        sens = results[label]["sensitivity"]
        total = results[label]["total"]
        if not np.isnan(sens):
            print(f"{label:<20} {sens:.3f}           {total}")
        else:
            print(f"{label:<20} {'N/A':<15} {total}")
    
    return results


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 7: DECISION CURVE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def decision_curve_analysis(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    disease_name: str,
    save_path: Path,
    threshold_range: Tuple[float, float] = (0.01, 0.99),
    n_thresholds: int = 100
) -> dict:
    """Decision Curve Analysis - shows clinical utility.
    
    SECRET WEAPON: Demonstrates net benefit vs "treat all" or "treat none".
    
    Scientific Basis:
    - Vickers & Elkin, "Decision Curve Analysis: A Novel Method for Evaluating 
      Prediction Models" (Medical Decision Making, 2006)
    - Standard in clinical prediction model evaluation
    
    Args:
        y_true: True labels
        y_scores: Predicted probabilities
        disease_name: Name of disease
        save_path: Path to save plot
        threshold_range: Range of probability thresholds to test
        n_thresholds: Number of thresholds to test
    
    Returns:
        Dict with net benefit curves
    """
    thresholds = np.linspace(threshold_range[0], threshold_range[1], n_thresholds)
    
    prevalence = y_true.mean()
    n = len(y_true)
    
    # Net benefit for each threshold
    net_benefit_model = []
    net_benefit_all = []
    net_benefit_none = []
    
    for pt in thresholds:
        # Model strategy: Treat if predicted probability ≥ pt
        y_pred = (y_scores >= pt).astype(int)
        
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        
        # Net benefit = (TP - FP × (pt/(1-pt))) / n
        # Interpretation: Benefit of true positives minus cost of false positives
        net_benefit = (tp - fp * (pt / (1 - pt))) / n
        net_benefit_model.append(net_benefit)
        
        # "Treat all" strategy
        # TP = all positives, FP = all negatives
        tp_all = y_true.sum()
        fp_all = (1 - y_true).sum()
        net_benefit_all.append((tp_all - fp_all * (pt / (1 - pt))) / n)
        
        # "Treat none" strategy
        net_benefit_none.append(0)
    
    # Plot decision curve
    plt.figure(figsize=(10, 6))
    
    plt.plot(thresholds, net_benefit_model, linewidth=2, label='Model', color='blue')
    plt.plot(thresholds, net_benefit_all, linewidth=2, label='Treat All', 
             color='red', linestyle='--')
    plt.plot(thresholds, net_benefit_none, linewidth=2, label='Treat None', 
             color='black', linestyle=':')
    
    plt.xlabel('Probability Threshold', fontsize=12)
    plt.ylabel('Net Benefit', fontsize=12)
    plt.title(f'Decision Curve Analysis - {disease_name}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.xlim(threshold_range)
    
    # Add interpretation text
    max_net_benefit = max(net_benefit_model)
    max_threshold = thresholds[np.argmax(net_benefit_model)]
    
    plt.text(0.5, 0.95, 
             f'Max Net Benefit: {max_net_benefit:.3f} at threshold {max_threshold:.2f}',
             transform=plt.gca().transAxes,
             fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n{disease_name.upper()} - Decision Curve Analysis:")
    print(f"  Max net benefit: {max_net_benefit:.3f}")
    print(f"  Optimal threshold: {max_threshold:.2f}")
    print(f"  Prevalence: {prevalence:.3f}")
    print(f"✅ Decision curve saved: {save_path}")
    
    return {
        "thresholds": thresholds.tolist(),
        "net_benefit_model": net_benefit_model,
        "net_benefit_all": net_benefit_all,
        "max_net_benefit": float(max_net_benefit),
        "optimal_threshold": float(max_threshold),
        "prevalence": float(prevalence)
    }


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 10: POWER ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def power_analysis_auc_comparison(
    auc1: float,
    auc2: float,
    prevalence: float,
    alpha: float = 0.05,
    power: float = 0.80,
    correlation: float = 0.5
) -> dict:
    """Compute required sample size for AUC comparison.
    
    Scientific Basis:
    - Hanley & McNeil, "The Meaning and Use of the Area under a ROC Curve" (1982)
    - Jung (2024): "A practical non-parametric statistical test to compare ROC curves"
    
    Args:
        auc1: AUC of model 1
        auc2: AUC of model 2
        prevalence: Proportion of positive cases
        alpha: Type I error rate (default: 0.05)
        power: Desired power (1 - Type II error) (default: 0.80)
        correlation: Correlation between tests (default: 0.5)
    
    Returns:
        Dict with required sample size and power analysis details
    """
    # Approximate variance of AUC (Hanley & McNeil, 1982)
    def auc_variance(auc, n_pos, n_neg):
        q1 = auc / (2 - auc)
        q2 = 2 * auc**2 / (1 + auc)
        
        var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc**2) + 
               (n_neg - 1) * (q2 - auc**2)) / (n_pos * n_neg)
        
        return var
    
    # Z-scores
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    
    # Estimate sample size (iterative)
    delta_auc = abs(auc1 - auc2)
    
    if delta_auc < 0.01:
        print("⚠️ WARNING: Very small AUC difference (<0.01) requires very large sample size")
    
    # Initial guess
    n_total = 100
    
    for _ in range(100):  # Iterative refinement
        n_pos = int(n_total * prevalence)
        n_neg = n_total - n_pos
        
        if n_pos < 10 or n_neg < 10:
            n_total += 100
            continue
        
        var1 = auc_variance(auc1, n_pos, n_neg)
        var2 = auc_variance(auc2, n_pos, n_neg)
        
        # Covariance (approximate using correlation)
        cov = correlation * np.sqrt(var1 * var2)
        
        # Required sample size
        var_diff = var1 + var2 - 2 * cov
        
        if var_diff <= 0:
            n_total += 100
            continue
        
        n_required = (z_alpha + z_beta)**2 * var_diff / (delta_auc**2)
        
        if abs(n_required - n_total) < 10:
            break
        
        n_total = int(n_required * 1.1)  # Add 10% margin
    
    n_required_final = int(np.ceil(n_total))
    n_pos_final = int(n_required_final * prevalence)
    n_neg_final = n_required_final - n_pos_final
    
    print(f"\nPower Analysis (Hanley & McNeil Method):")
    print(f"  AUC1: {auc1:.3f}")
    print(f"  AUC2: {auc2:.3f}")
    print(f"  ΔAUC: {delta_auc:.3f}")
    print(f"  Prevalence: {prevalence:.2%}")
    print(f"  Desired power: {power:.2%}")
    print(f"  Significance level: {alpha:.3f}")
    print(f"  Correlation: {correlation:.2f}")
    print(f"\n  Required sample size: {n_required_final}")
    print(f"    Positive cases: {n_pos_final}")
    print(f"    Negative cases: {n_neg_final}")
    
    return {
        "n_required": n_required_final,
        "n_pos": n_pos_final,
        "n_neg": n_neg_final,
        "auc1": auc1,
        "auc2": auc2,
        "delta_auc": delta_auc,
        "prevalence": prevalence,
        "power": power,
        "alpha": alpha,
        "correlation": correlation,
        "method": "Hanley & McNeil (1982)"
    }


if __name__ == "__main__":
    print("NeuroX Rejection-Proof Evaluation System - Phase 5-7")
    print("="*80)
    print("\nThis module provides:")
    print("  1. Lesion-level IoU matching")
    print("  2. Sensitivity vs lesion size curve")
    print("  3. Decision Curve Analysis (clinical utility)")
    print("  4. Power analysis (Hanley & McNeil)")
    print("\nImport and use in your evaluation pipeline.")
