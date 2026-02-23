
import sys
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from neurox_adaptive import load_model, load_and_preprocess_nifti, automatic_disease_detection, perform_segmentation, compute_lesion_metrics

def evaluate_on_file(file_path):
    print(f"🔍 Evaluating model on: {file_path}")
    
    # 1. Load Model
    model = load_model()
    if model is None:
        print("❌ Model loading failed.")
        return

    # 2. Preprocess
    try:
        roi_tensor, original_data, roi_metadata, affine, spacing = load_and_preprocess_nifti(file_path)
        print(f"✅ Preprocessing successful. Tensor shape: {roi_tensor.shape}")
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        return

    # 3. Detection
    try:
        detection = automatic_disease_detection(model, roi_tensor)
        print("✅ Detection Results:")
        for disease in ["tumor", "stroke", "alzheimer"]:
            prob = detection["probabilities"].get(disease, 0)
            unc = detection["uncertainties"].get(disease, 0)
            print(f"   - {disease.capitalize()}: Prob={prob:.4f}, Uncertainty={unc:.4f}")
        print(f"   - Detected: {detection['detected_diseases']}")
    except Exception as e:
        print(f"❌ Detection failed: {e}")
        return

    # 4. Segmentation (if applicable)
    detected = detection["detected_diseases"]
    seg_diseases = [d for d in detected if d in ["tumor", "stroke"]]
    
    if seg_diseases:
        try:
            seg_results = perform_segmentation(model, roi_tensor, seg_diseases)
            for disease, (probs, binary) in seg_results.items():
                print(f"✅ Segmentation successful for {disease}. Binary sum: {binary.sum()}")
                
                # Metrics
                metrics = compute_lesion_metrics(binary, affine)
                if metrics:
                    print(f"   - Volume: {metrics['volume_mm3']:.2f} mm3")
                    print(f"   - Centroid: {metrics['centroid_mm']}")
                else:
                    print(f"   - No lesion metrics (empty mask in world space)")
        except Exception as e:
            print(f"❌ Segmentation failed: {e}")
    else:
        print("ℹ️ No segmentable diseases detected.")

if __name__ == "__main__":
    file_to_test = "brain3.nii.gz"
    if Path(file_to_test).exists():
        evaluate_on_file(file_to_test)
    else:
        print(f"❌ File {file_to_test} not found.")
