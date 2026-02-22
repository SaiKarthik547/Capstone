"""
NeuroX Evaluation Integration Test
Tests all 4 evaluation modules using synthetic data.
Run from the evaluation folder:  py run_evaluation_test.py
"""

import sys
import os
import tempfile
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from calibration import (
    TemperatureScaling,
    compute_expected_calibration_error,
    report_calibration_metrics,
    compute_roc_operating_points,
    cost_ratio_sensitivity_analysis,
)
from statistical_rigor import (
    compute_lesion_level_iou_matching,
    sensitivity_vs_lesion_size_curve,
    decision_curve_analysis,
    power_analysis_auc_comparison,
)
from nested_cv import verify_multi_label_training

PASS = []
FAIL = []

TMP = Path(tempfile.mkdtemp())  # temp dir for plot files


def run(name, fn):
    try:
        fn()
        print(f"  ✅ PASS  {name}")
        PASS.append(name)
    except Exception as e:
        import traceback
        print(f"  ❌ FAIL  {name}: {e}")
        traceback.print_exc()
        FAIL.append(name)


# ── Synthetic data ───────────────────────────────────────────────────────────
np.random.seed(42)
N = 200
DISEASE_NAMES = ["tumor", "stroke", "alzheimer"]

labels_dict = {d: torch.from_numpy((np.random.rand(N) > 0.6).astype(np.float32))
               for d in DISEASE_NAMES}
logits_dict = {d: torch.from_numpy(np.random.randn(N).astype(np.float32))
               for d in DISEASE_NAMES}
probs_dict = {d: torch.sigmoid(logits_dict[d]).numpy() for d in DISEASE_NAMES}


# ── TEST 1: Multi-label BCE verification ─────────────────────────────────────
print("\n" + "="*60)
print("TEST 1 — Multi-label BCE / dataset verification")
print("="*60)

class _FakeDataset:
    """Flat-key dataset matching nested_cv.py's expected keys."""
    def __init__(self):
        self.tumor = labels_dict["tumor"].numpy()
        self.stroke = labels_dict["stroke"].numpy()
        self.alzheimer = labels_dict["alzheimer"].numpy()
    def __len__(self): return N
    def __getitem__(self, i):
        return {
            "tumor_presence":     torch.tensor([self.tumor[i]]),
            "stroke_presence":    torch.tensor([self.stroke[i]]),
            "alzheimer_presence": torch.tensor([self.alzheimer[i]]),
        }

class _FakeModel:
    """Minimal model stub exposing presence_heads and seg_decoders."""
    class _DictLike:
        def keys(self): return ["tumor", "stroke"]
    presence_heads = _DictLike()
    seg_decoders   = _DictLike()

def test_multilabel():
    ds = _FakeDataset()
    model = _FakeModel()
    result = verify_multi_label_training(model, ds)
    assert isinstance(result, dict), "Expected dict from verify_multi_label_training"
    assert "multi_label_count" in result
    assert "disease_counts" in result
    print(f"    Multi-label samples: {result['multi_label_count']}/{result['total_checked']}")
    print(f"    Disease counts: { {k: v for k,v in result['disease_counts'].items()} }")

run("Multi-label BCE verification", test_multilabel)


# ── TEST 2: Temperature Scaling ───────────────────────────────────────────────
print("\n" + "="*60)
print("TEST 2 — Temperature scaling calibration")
print("="*60)

def test_temperature_scaling():
    ts = TemperatureScaling(num_diseases=3)
    temps = ts.fit(logits_dict, labels_dict, DISEASE_NAMES)
    assert set(temps.keys()) == set(DISEASE_NAMES), "Missing disease in temperatures"
    for d, t in temps.items():
        assert t > 0, f"Temperature must be > 0, got {t} for {d}"
    print(f"    Optimal temps: { {d: round(t,3) for d,t in temps.items()} }")

run("Temperature scaling", test_temperature_scaling)


# ── TEST 3: ECE + Calibration metrics ────────────────────────────────────────
print("\n" + "="*60)
print("TEST 3 — ECE + calibration metrics")
print("="*60)

def test_ece():
    for d in DISEASE_NAMES:
        y_true = labels_dict[d].numpy()
        y_prob = probs_dict[d]
        ece = compute_expected_calibration_error(y_true, y_prob, n_bins=10)
        assert 0.0 <= ece <= 1.0, f"ECE out of range: {ece}"
        # report_calibration_metrics(y_true, y_prob_before, y_prob_after, disease_name)
        metrics = report_calibration_metrics(y_true, y_prob, y_prob, d)
        assert "brier_before" in metrics
        print(f"    {d}: ECE={ece:.4f}  Brier={metrics['brier_before']:.4f}")

run("ECE + calibration metrics", test_ece)


# ── TEST 4: ROC operating points ─────────────────────────────────────────────
print("\n" + "="*60)
print("TEST 4 — ROC operating points")
print("="*60)

def test_roc_op():
    for d in DISEASE_NAMES:
        y_true = labels_dict[d].numpy()
        y_prob = probs_dict[d]
        # compute_roc_operating_points(y_true, y_scores, disease_name) -> dict
        op = compute_roc_operating_points(y_true, y_prob, d)
        assert isinstance(op, dict), f"Expected dict, got {type(op)}"
        # Print whatever keys came back
        print(f"    {d}: {list(op.keys())[:4]}...")

run("ROC operating points", test_roc_op)


# ── TEST 5: Lesion-level IoU matching ────────────────────────────────────────
print("\n" + "="*60)
print("TEST 5 — Lesion-level IoU matching")
print("="*60)

def test_iou_matching():
    vol  = np.zeros((64, 64, 64), dtype=np.uint8)
    pred = np.zeros_like(vol)
    # A synthetic lesion cube in GT
    vol[10:20, 10:20, 10:20] = 1
    # Prediction overlaps significantly
    pred[12:22, 12:22, 12:22] = 1
    # correct kwarg: iou_threshold (not iou_thresh)
    result = compute_lesion_level_iou_matching(pred, vol, iou_threshold=0.1)
    assert "tp" in result, f"Expected 'tp' key, got: {list(result.keys())}"
    print(f"    TP={result['tp']}  FP={result['fp']}  FN={result['fn']}")
    print(f"    n_gt={result['n_gt_lesions']}  n_pred={result['n_pred_lesions']}")
    assert result["tp"] >= 1, f"Should detect at least 1 TP lesion"

run("Lesion-level IoU matching", test_iou_matching)


# ── TEST 6: Sensitivity vs lesion size ───────────────────────────────────────
print("\n" + "="*60)
print("TEST 6 — Sensitivity vs lesion size curve")
print("="*60)

def test_sensitivity_vs_size():
    preds = []
    gts   = []
    for _ in range(8):
        size = np.random.randint(6, 16)
        gt   = np.zeros((64, 64, 64), dtype=np.uint8)
        pred = np.zeros_like(gt)
        s = np.random.randint(5, 45)
        gt[s:s+size, s:s+size, s:s+size] = 1
        off = np.random.randint(0, 3)
        e = min(s+off+size, 64)
        pred[s+off:e, s+off:e, s+off:e] = 1
        # sensitivity_vs_lesion_size_curve expects list of dicts with 'lesion_mask' key
        preds.append({"lesion_mask": pred})
        gts.append({"lesion_mask": gt})
    # signature: (predictions, ground_truth, voxel_spacing, iou_threshold)
    result = sensitivity_vs_lesion_size_curve(preds, gts, voxel_spacing=(1.0, 1.0, 1.0))
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    print(f"    Size bins evaluated: {len(result)}")

run("Sensitivity vs lesion size", test_sensitivity_vs_size)


# ── TEST 7: Decision curve analysis ──────────────────────────────────────────
print("\n" + "="*60)
print("TEST 7 — Decision curve analysis")
print("="*60)

def test_dca():
    for d in DISEASE_NAMES:
        y_true = labels_dict[d].numpy()
        y_prob = probs_dict[d]
        save_path = TMP / f"{d}_decision_curve.png"
        # signature: (y_true, y_scores, disease_name, save_path, ...)
        result = decision_curve_analysis(y_true, y_prob, d, save_path)
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        assert "thresholds" in result
        print(f"    {d}: {len(result['thresholds'])} threshold pts  "
              f"max_net_benefit={result['max_net_benefit']:.3f}")

run("Decision curve analysis", test_dca)


# ── TEST 8: Power analysis ────────────────────────────────────────────────────
print("\n" + "="*60)
print("TEST 8 — Power analysis (Hanley & McNeil)")
print("="*60)

def test_power():
    result = power_analysis_auc_comparison(
        auc1=0.85, auc2=0.70, prevalence=0.4, alpha=0.05, power=0.80
    )
    assert "n_required" in result, f"Expected n_required, got: {list(result.keys())}"
    print(f"    Required N={result['n_required']}  pos={result['n_pos']}  neg={result['n_neg']}")

run("Power analysis", test_power)


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"RESULTS: {len(PASS)}/{len(PASS)+len(FAIL)} tests passed")
print("="*60)
for t in PASS: print(f"  ✅  {t}")
for t in FAIL: print(f"  ❌  {t}")

if FAIL:
    sys.exit(1)
