
import torch
from pathlib import Path

def extract_best_metrics(model_path):
    if not Path(model_path).exists():
        print(f"❌ Model {model_path} not found.")
        return

    checkpoint = torch.load(model_path, map_location="cpu")
    if "metrics" not in checkpoint:
        print("❌ No metrics found in checkpoint.")
        return

    metrics = checkpoint["metrics"]
    print("--- BEST MODEL METRICS (from Training History) ---")
    
    if "tumor_et" in metrics and metrics["tumor_et"]:
        print(f"Tumor ET Dice (Peak): {max(metrics['tumor_et']):.4f}")
    if "tumor_mean" in metrics and metrics["tumor_mean"]:
        print(f"Tumor Mean Dice (Peak): {max(metrics['tumor_mean']):.4f}")
    if "stroke_dice" in metrics and metrics["stroke_dice"]:
        print(f"Stroke Dice (Peak): {max(metrics['stroke_dice']):.4f}")
    if "alz_auc" in metrics and metrics["alz_auc"]:
        print(f"Alzheimer AUC (Peak): {max(metrics['alz_auc']):.4f}")
    if "alz_accuracy" in metrics and metrics["alz_accuracy"]:
        print(f"Alzheimer Accuracy (Peak): {max(metrics['alz_accuracy']):.4f}")
    if "alz_f1" in metrics and metrics["alz_f1"]:
        print(f"Alzheimer F1 (Peak): {max(metrics['alz_f1']):.4f}")
    
    epochs = len(metrics.get("epoch", []))
    print(f"Training Duration: {epochs} epochs")

if __name__ == "__main__":
    extract_best_metrics("checkpoints/neurox_model.pth")
