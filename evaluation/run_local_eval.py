import os
import torch
import nibabel as nib
import numpy as np
from pathlib import Path
import neurox_adaptive as nx

def evaluate_local_files():
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "checkpoints/neurox_model.pth"
    if not os.path.exists(model_path):
        print(f"Error: Model {model_path} not found")
        return
    
    # Manually load
    model = nx.NeuroXMultiDisease(in_channels=2)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("model_state", checkpoint.get("model", checkpoint))
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    
    test_files = ["alzeimers_scan.nii.gz", "brain3.nii.gz", "brain_flair.nii.gz", "stroke brain.nii.gz"]
    print("| Scan Identifier | Pathology Detection | Probabilities | Uncertainty |")
    print("| :--- | :--- | :--- | :--- |")
    
    for filename in test_files:
        if not os.path.exists(filename): continue
        try:
            # FIX: Unpack 6 values
            roi_alz, roi_standard, roi_orig, roi_meta, aff, spacing = nx.load_and_preprocess_nifti(filename)
            
            x_alz = roi_alz.to(device).unsqueeze(0)
            x_seg = torch.cat([roi_standard.to(device).unsqueeze(0)]*2, dim=1)
            
            with torch.no_grad():
                out_alz = model(x_alz, active_presence=["alzheimer"])
                out_seg = model(x_seg, active_presence=["tumor", "stroke"])
                t_l, t_v = out_seg["presence"]["tumor"]
                s_l, s_v = out_seg["presence"]["stroke"]
                a_l, a_v = out_alz["presence"]["alzheimer"], out_alz["alzheimer_log_var"]
                
                tp, sp, ap = torch.sigmoid(t_l).item(), torch.sigmoid(s_l).item(), torch.sigmoid(a_l).item()
                tu, su, au = torch.exp(t_v).item(), torch.exp(s_v).item(), torch.exp(a_v).item()
                
                det, pr, un = [], [], []
                if tp > 0.4: det.append("Tumor"); pr.append(f"T:{tp:.2f}"); un.append(f"T:{tu:.2f}")
                if sp > 0.4: det.append("Stroke"); pr.append(f"S:{sp:.2f}"); un.append(f"S:{su:.2f}")
                if ap > 0.4: det.append("Alzheimer"); pr.append(f"A:{ap:.2f}"); un.append(f"A:{au:.2f}")
                
                if not det: det, pr, un = ["Normal"], ["-"], ["-"]
                print(f"| {filename} | {', '.join(det)} | {', '.join(pr)} | {', '.join(un)} |")
        except Exception as e:
            print(f"| {filename} | Error: {str(e)} | - | - |")

if __name__ == "__main__":
    evaluate_local_files()
