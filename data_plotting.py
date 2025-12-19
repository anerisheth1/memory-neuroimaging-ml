import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import random
import os

def plot_random_neurosynth_maps_grid(memory_dir, nonmemory_dir, n_examples=5):
    """
    Loads and plots a random selection of memory and non-memory NeuroSynth maps
    in a 2-row, 3-column grid (3 images per row).

    Args:
        memory_dir (str): Path to the directory containing memory NIfTI files.
        nonmemory_dir (str): Path to the directory containing non-memory NIfTI files.
        n_examples (int): The number of examples of each type to plot (default is 3, total of 6 maps).
    """
    
    # 1. Get the list of all files
    try:
        all_mem_files = [os.path.join(memory_dir, f) for f in os.listdir(memory_dir) if f.endswith('.nii.gz')]
        all_non_mem_files = [os.path.join(nonmemory_dir, f) for f in os.listdir(nonmemory_dir) if f.endswith('.nii.gz')]
    except FileNotFoundError as e:
        print(f"Error: One of the directories was not found. Please check paths. {e}")
        return
    
    if not all_mem_files or not all_non_mem_files:
        print("Error: One or both directories are empty or contain no NIfTI files.")
        return

    # 2. Select random examples
    num_to_plot = min(n_examples, len(all_mem_files), len(all_non_mem_files))
    
    # Select 3 of each type
    mem_examples = random.sample(all_mem_files, num_to_plot)
    non_examples = random.sample(all_non_mem_files, num_to_plot)
    
    # Combine the lists and define metadata for plotting
    plot_data = []
    for f in mem_examples:
        plot_data.append({'file': f, 'type': 'Memory', 'cmap': 'Reds'})
    for f in non_examples:
        plot_data.append({'file': f, 'type': 'Non-Memory', 'cmap': 'Blues'})

    # 3. Setup Plot
    N_ROWS = 2
    N_COLS = 3
    
    plt.figure(figsize=(15, 8))
    plt.suptitle("Example NeuroSynth Maps (Middle Slice) - 3 Images per Row", fontsize=16, y=1.02)
    
    # Get the middle slice index 
    sample_img = nib.load(plot_data[0]['file'])
    brain_shape = sample_img.get_fdata().shape
    mid_slice = brain_shape[2] // 2
    
    # 4. Plotting data
    for i, item in enumerate(plot_data):

        plt.subplot(N_ROWS, N_COLS, i + 1)
        data = nib.load(item['file']).get_fdata()
        plt.imshow(np.rot90(data[:, :, mid_slice]), cmap=item['cmap'])
        map_name = os.path.basename(item['file']).split('_association-test')[0].replace('_', ' ').title()
        plt.title(f"{item['type']}:\n{map_name}", fontsize=10)
        plt.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

if __name__ == "__main__":
    
    MEMORY_DIR = 'data/memory_maps'
    NONMEMORY_DIR = 'data/nonmemory_maps'

    plot_random_neurosynth_maps_grid(MEMORY_DIR, NONMEMORY_DIR, n_examples=3)