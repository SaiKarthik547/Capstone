
import sys
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from neurox_adaptive import load_model, load_and_preprocess_nifti, automatic_disease_detection, perform_segmentation, compute_lesion_metrics

def evaluate_on_file(file_path, model):
    print(f"\n🔍 EVALUATING: {file_path}")
    
    # 2. Preprocess
    try:
        roi_tensor, original_data, roi_metadata, affine, spacing = load_and_preprocess_nifti(file_path)
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
            status = "DETECTED" if disease in detection["detected_diseases"] else "NEGATIVE"
            print(f"   - {disease.capitalize():<10}: Prob={prob:.4f}, Uncertainty={unc:.4f} -> {status}")
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
                # Metrics
                metrics = compute_lesion_metrics(binary, affine)
                if metrics:
                    print(f"✅ {disease.capitalize()} Segmented: {metrics['volume_mm3']:.2f} mm3")
                else:
                    print(f"⚠️ {disease.capitalize()} segmented but empty in world space.")
        except Exception as e:
            print(f"❌ Segmentation failed: {e}")
    else:
        print("ℹ️ No segmentable pathology detected.")

def main():
    print("="*60)
    print("NEUROX 80-EPOCH MODEL - MULTI-CASE EVALUATION")
    print("="*60)
    
    model = load_model()
    if model is None:
        print("❌ Model loading failed.")
        return

    test_files = ["brain3.nii.gz", "brain_flair.nii.gz", "stroke brain.nii.gz"]
    
    for f in test_files:
        if Path(f).exists():
            evaluate_on_file(f, model)
        else:
            print(f"⚠️ File {f} not found.")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
