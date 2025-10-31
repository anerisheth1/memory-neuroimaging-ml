import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Load the NIfTI file
img = nib.load("data/functional_connectivity_-46_0_0.nii")
data = img.get_fdata()

print("Data shape:", data.shape)

# Select a slice (middle slice along the z-axis, for example)
slice_z = 59  # can adjust this depending on your file
slice_data = data[:, :, slice_z]

# Display with a perceptually appropriate colormap
plt.imshow(np.rot90(slice_data), cmap="gray")
plt.title(f"Brain slice at z = {slice_z}")
plt.axis("off")
plt.show()
