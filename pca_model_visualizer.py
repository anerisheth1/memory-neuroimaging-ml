from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import nibabel as nib

from data_loader import load_data

# Assume data_loader.py exists and load_data() works as intended
# from data_loader import load_data 


memory_dir = "data/memory_maps/"
nonmemory_dir = "data/nonmemory_maps/"

# File paths are only needed here to get a sample image for the visualization
memory_files = sorted(glob.glob(os.path.join(memory_dir, "*.nii*")))
nonmemory_files = sorted(glob.glob(os.path.join(nonmemory_dir, "*.nii*")))

def pca_analysis(n_components=8):
    """Performs PCA, plots explained variance and 2D projection, returns fitted PCA object and reduced data."""
    X, y = load_data()

    pca_model = PCA(n_components=n_components)
    X_pca = pca_model.fit_transform(X)

    print("Original shape:", X.shape)
    print("PCA-reduced shape:", X_pca.shape)
    
    
    # Image of cumulative explained variance plot

    plt.figure(figsize=(8,5))
    plt.plot(np.cumsum(pca_model.explained_variance_ratio_), marker='o')
    plt.xlabel("Number of PCA components")
    plt.ylabel("Cumulative explained variance")
    plt.title("Explained Variance by PCA Components")
    plt.grid(alpha=0.3)
    plt.show()

    plt.figure(figsize=(7,6))

    # Only use the first two PCA dimensions
    pc1 = X_pca[:,0]
    pc2 = X_pca[:,1]

    plt.scatter(pc1[y==1], pc2[y==1], c='red', label='Memory', s=80)
    plt.scatter(pc1[y==0], pc2[y==0], c='blue', label='Non-Memory', s=80)

    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title("PCA Projection of NeuroSynth Maps")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    print("\nExplained Variance Ratios:")
    # Print up to 10 components or all if less than 10
    n_print = min(10, pca_model.n_components_) 
    for i, var in enumerate(pca_model.explained_variance_ratio_[:n_print]):
        print(f"PC{i+1}: {var:.4f}")
        
    return pca_model, X_pca

def visualize_pca(pca_model):
    """Visualizes the first PCA component as a 3D brain map slice."""

    # Ensure memory_files is not empty
    if not memory_files:
        print("Error: No NIfTI files found in memory directory to load a sample image.")
        return
        
    # Get a sample image for shape/affine data
    try:
        sample_img = nib.load(memory_files[0])
        sample_data = sample_img.get_fdata()
    except Exception as e:
        print(f"Error loading sample NIfTI file: {e}")
        return

    brain_shape = sample_data.shape   # e.g., (91, 109, 91)
    print("\nBrain shape:", brain_shape)

    # Check if the PCA model was fitted
    if not hasattr(pca_model, 'components_'):
        print("Error: The provided PCA model has not been fitted.")
        return

    # Components are stored as (n_components, n_voxels). We reshape each row.
    components_3d = []
    
    # We only need to check against the actual number of components fitted
    n_components_to_vis = min(pca_model.n_components, len(pca_model.components_))
    
    for i in range(n_components_to_vis):
        comp_flat = pca_model.components_[i]          # shape: (n_voxels,)
        comp_3d = comp_flat.reshape(brain_shape) # reshape to original NIfTI shape
        components_3d.append(comp_3d)

    pc_idx = 1    # We will visualize PC 1

    if len(components_3d) < 1:
        print("Error: No PCA components available for visualization.")
        return

    pc_map = components_3d[pc_idx]
    

    plt.figure(figsize=(6,5))
    # Using np.rot90 to orient the image correctly for display
    plt.imshow(np.rot90(pc_map[:, :, brain_shape[2]//2]), cmap='RdBu_r')
    plt.title(f"PCA Component {pc_idx+1} (middle slice)")
    plt.colorbar(shrink=0.7)
    plt.axis("off")
    plt.show()
    
    orig = sample_data   # The data from your sample NIfTI file

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.imshow(np.rot90(orig[:, :, brain_shape[2]//2]), cmap="gray")
    plt.title("Original NeuroSynth Map (Middle Slice) \n Amnestic Association")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(np.rot90(pc_map[:, :, brain_shape[2]//2]), cmap="RdBu_r")
    plt.title("PCA Component 1 (Middle Slice)/n Reshaped to Same Brain Dim")
    plt.axis("off")

    plt.show()


# --- Execution Block ---
# Run the functions to see the result
if __name__ == "__main__":
    # 1. Run PCA
    fitted_pca_model, X_reduced = pca_analysis()

    # 2. Visualize the results using the fitted model
    visualize_pca(fitted_pca_model)