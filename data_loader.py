import nibabel as nib
import numpy as np
import glob
import os

def load_data():
    memory_dir = "data/memory_maps/"
    nonmemory_dir = "data/nonmemory_maps/"

    memory_files = sorted(glob.glob(os.path.join(memory_dir, "*.nii*")))
    nonmemory_files = sorted(glob.glob(os.path.join(nonmemory_dir, "*.nii*")))

    print("Memory maps found:", len(memory_files))
    print("Non-memory maps found:", len(nonmemory_files))

    X = []
    y = []
    label_names = []

    # Load memory maps → label = 1
    for f in memory_files:
        img = nib.load(f).get_fdata()
        vec = img.flatten().astype(np.float32)
        X.append(vec)
        y.append(1)
        label_names.append(os.path.basename(f))

    # Load non-memory maps → label = 0
    for f in nonmemory_files:
        img = nib.load(f).get_fdata()
        vec = img.flatten().astype(np.float32)
        X.append(vec)
        y.append(0)
        label_names.append(os.path.basename(f))

    X = np.vstack(X)
    y = np.array(y)

    print("Final X shape:", X.shape)
    print("Final y shape:", y)
    print("Labels loaded:", label_names)

    return X, y