import torch
from pathlib import Path

def debug_metrics(model_path):
    if not Path(model_path).exists():
        print(f"File {model_path} not found")
        return
    checkpoint = torch.load(model_path, map_location="cpu")
    if "metrics" in checkpoint:
        print(f"Keys found: {list(checkpoint['metrics'].keys())}")
        for k, v in checkpoint['metrics'].items():
            if isinstance(v, list):
                print(f"  {k}: length {len(v)}")
            else:
                print(f"  {k}: value {v}")
    else:
        print("No 'metrics' key in checkpoint")
        print(f"All keys: {list(checkpoint.keys())}")

if __name__ == "__main__":
    debug_metrics("checkpoints/neurox_model.pth")
