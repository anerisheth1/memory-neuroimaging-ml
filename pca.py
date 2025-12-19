from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import nibabel as nib

from data_loader import load_data

memory_dir = "data/memory_maps/"
nonmemory_dir = "data/nonmemory_maps/"

memory_files = sorted(glob.glob(os.path.join(memory_dir, "*.nii*")))
nonmemory_files = sorted(glob.glob(os.path.join(nonmemory_dir, "*.nii*")))

def pca(n_components = 5):
    X, y = load_data()

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    print("Original shape:", X.shape)
    print("PCA-reduced shape:", X_pca.shape)

    plt.figure(figsize=(8,5))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
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

    for i, var in enumerate(pca.explained_variance_ratio_[:10]):
        print(f"PC{i+1}: {var:.4f}")
    return X_pca

def visualize_pca():

    sample_img = nib.load(memory_files[0])
    sample_data = sample_img.get_fdata()
    brain_shape = sample_data.shape   # e.g., (91, 109, 91)
    print("Brain shape:", brain_shape)

    components_3d = []

    for i in range(pca.n_components):
        comp_flat = pca.components_[i]          # shape: (n_voxels,)
        comp_3d = comp_flat.reshape(brain_shape) # reshape to original NIfTI shape
        components_3d.append(comp_3d)

    pc_idx = 0    # PC1

    pc_map = components_3d[pc_idx]

    plt.figure(figsize=(6,5))
    plt.imshow(np.rot90(pc_map[:, :, brain_shape[2]//2]), cmap='RdBu_r')
    plt.title(f"PCA Component {pc_idx+1} (middle slice)")
    plt.colorbar(shrink=0.7)
    plt.axis("off")
    plt.show()
    
    orig = sample_data   # from earlier: your memory_files[0]

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.imshow(np.rot90(orig[:, :, brain_shape[2]//2]), cmap="gray")
    plt.title("Original NeuroSynth Map")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(np.rot90(pc_map[:, :, brain_shape[2]//2]), cmap="RdBu_r")
    plt.title("PCA Component 1")
    plt.axis("off")

    plt.show()


    # thresh = np.percentile(np.abs(pc_map), 95)
    # pc_thresh = np.where(np.abs(pc_map) > thresh, pc_map, 0)
    # plt.imshow(np.rot90(pc_thresh[:, :, brain_shape[2]//2]), cmap='RdBu_r')


