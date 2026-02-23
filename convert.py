import sys
import os
import argparse
import nibabel as nib

def convert_nii_to_niigz(input_path: str):
    """
    Converts a .nii file to .nii.gz format.
    
    Args:
        input_path (str): Path to the input .nii file.
    """
    if not input_path.lower().endswith('.nii'):
        # print(f"Skipping {input_path}: Not a .nii file")
        return

    output_path = input_path + ".gz"
    
    # Check if output already exists
    if os.path.exists(output_path):
        print(f"Skipping {input_path}: {output_path} already exists")
        return

    try:
        print(f"Converting: {input_path} -> {output_path}")
        img = nib.load(input_path)
        nib.save(img, output_path)
        print(f"✅ Success: Saved to {output_path}")
    except Exception as e:
        print(f"❌ Error converting {input_path}: {e}")

def main():
    # If arguments provided, use them. Otherwise default to current directory.
    if len(sys.argv) > 1:
        target_path = sys.argv[1]
    else:
        target_path = "."
    
    if os.path.isfile(target_path):
        convert_nii_to_niigz(target_path)
    elif os.path.isdir(target_path):
        print(f"Scanning directory: {os.path.abspath(target_path)}")
        files = [f for f in os.listdir(target_path) if f.lower().endswith('.nii')]
        
        if not files:
            print("No .nii files found in this directory.")
            return

        print(f"Found {len(files)} .nii files.")
        for file in files:
            full_path = os.path.join(target_path, file)
            convert_nii_to_niigz(full_path)
    else:
        print(f"Error: Path not found: {target_path}")

if __name__ == "__main__":
    main()
