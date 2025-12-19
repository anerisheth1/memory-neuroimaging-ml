import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import tensorflow as tf 
from pca import memory_files  # just to get a sample image shape

def saliency_brain_map(clf, pca_model, save_path=None, title="Saliency Map"):
    """
    Reconstructs classifier weights (SVM or Logistic Regression) back to brain voxel space.
    Args:
        clf: trained classifier with coef_ (SVM, Logistic Regression)
        pca_model: fitted PCA model with components_
        save_path: optional file path to save NIfTI output
        title: plot title
    """

    # ---- STEP 1: Extract linear weights from the classifier ----
    weights_pca = clf.coef_.flatten()  # shape: (n_components,)

    # ---- STEP 2: Map weights from PCA space back to voxel space ----
    voxel_weights = np.dot(weights_pca, pca_model.components_)  # shape: (n_voxels,)

    # ---- STEP 3: Reshape to original 3D brain volume ----
    sample_img = nib.load(memory_files[0])
    sample_data = sample_img.get_fdata()
    brain_shape = sample_data.shape
    voxel_map_3d = voxel_weights.reshape(brain_shape)

    # ---- STEP 4: Visualization (middle axial slice) ----
    mid = brain_shape[2] // 2

    plt.figure(figsize=(6,5))
    plt.imshow(np.rot90(voxel_map_3d[:, :, mid]), cmap="RdBu_r")
    plt.colorbar(shrink=0.6, label="Classifier Weight")
    plt.title(f"{title}\n(red = memory, blue = non-memory)", fontsize=12)
    plt.axis("off")
    plt.show()

    # ---- STEP 5: Optional — Save as NIfTI file ----
    if save_path:
        img = nib.Nifti1Image(voxel_map_3d, sample_img.affine)
        nib.save(img, save_path)
        print(f"Saved saliency map to: {save_path}")

    return voxel_map_3d


def saliency_full_voxel_map(clf, save_path=None, title="L1 Logistic Regression Saliency Map"):
    """
    Creates a voxel-wise saliency map from L1 logistic regression trained on full voxel data.
    """

    # ---- STEP 1: Extract voxel weights (already in voxel space) ----
    voxel_weights = clf.coef_.flatten()

    # ---- STEP 2: Load sample brain to get shape and affine ----
    sample_img = nib.load(memory_files[0])
    sample_data = sample_img.get_fdata()
    brain_shape = sample_data.shape

    # ---- STEP 3: Reshape coefficients into brain volume ----
    voxel_map_3d = voxel_weights.reshape(brain_shape)

    # ---- STEP 4: Show middle axial slice ----
    mid = brain_shape[2] // 2
    plt.figure(figsize=(6,5))
    plt.imshow(np.rot90(voxel_map_3d[:, :, mid]), cmap="RdBu_r")
    plt.title(f"{title}\n(red = memory, blue = non-memory)", fontsize=11)
    plt.colorbar(shrink=0.6, label="Weight (L1)")
    plt.axis("off")
    plt.show()

    # ---- STEP 5: Optional save NIfTI ----
    if save_path:
        sal_map = nib.Nifti1Image(voxel_map_3d, sample_img.affine)
        nib.save(sal_map, save_path)
        print(f"Saved full-data saliency map to: {save_path}")

    return voxel_map_3d

def nn_saliency_map(model, sample_input, title="NN Saliency Map"):
    """
    Computes a gradient-based saliency map for a Keras model
    trained on voxel-wise brain data.
    
    Args:
        model: Trained Keras binary classifier
        sample_input: A single standardized input sample (1 x voxels)
    """

    # ---- STEP 1: Convert to tensor and watch it ----
    x = tf.convert_to_tensor(sample_input.reshape(1, -1), dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(x)
        y_pred = model(x)  # forward pass (probability of memory)

    # ---- STEP 2: Compute gradients of output wrt voxels ----
    grads = tape.gradient(y_pred, x).numpy().flatten()  # shape: (voxels,)

    # ---- STEP 3: Load brain shape (from any NeuroSynth file) ----
    sample_img = nib.load(memory_files[0])
    brain_shape = sample_img.get_fdata().shape

    # ---- STEP 4: Reshape gradient vector into 3D brain ----
    sal_map_3d = grads.reshape(brain_shape)

    # ---- STEP 5: Plot middle slice ----
    mid = brain_shape[2] // 2
    plt.figure(figsize=(6,5))
    plt.imshow(np.rot90(sal_map_3d[:, :, mid]), cmap="RdBu_r")
    plt.title(f"{title}\n(red = +memory, blue = +non-memory)", fontsize=11)
    plt.colorbar(shrink=0.6, label="Gradient")
    plt.axis("off")
    plt.show()

    return sal_map_3d
