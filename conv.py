import nibabel as nib
import os

# Your .nii file path
input_path = "C:\\Users\\karth\\OneDrive\\Desktop\\neurox\\brain2.nii"
output_path = "C:\\Users\\karth\\OneDrive\\Desktop\\neurox\\brain2.nii.gz"

# Load the .nii file
img = nib.load(input_path)

# Save it as .nii.gz
nib.save(img, output_path)

print(f"Converted: {output_path}")