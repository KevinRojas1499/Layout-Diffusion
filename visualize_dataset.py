import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.datasets import get_dataset

def plot_sample_from_dataset():
    dataset = get_dataset('euclidean_variable_length_toy', max_length=30)
    iterator = iter(dataset)
    sample = next(iterator)
    data = sample["data"]
    mask = sample["mask"]
    plot_sample(data, mask, 'dataset_visualization.png')

def plot_sample(data, mask, out_file_name):
    # Convert to numpy for plotting
    data_np = data.numpy()
    mask_np = mask.numpy()
    
    # Reshape 1D data for heatmap visualization if necessary
    if data_np.ndim == 1:
        data_viz = data_np[:, None]
        mask_viz = mask_np[:, None]
    else:
        data_viz = data_np
        mask_viz = mask_np
        if mask_viz.ndim == 1:
             mask_viz = np.tile(mask_viz[:, None], (1, data_viz.shape[1]))
    
    # Calculate valid length
    # Mask is True for valid tokens, False for padding
    valid_len = mask_np.sum()
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    max_len = data_np.shape[0]
    
    plt.subplot(1, 2, 1)
    # Fixed scale from -1 to max_len + 1
    # We want to hide the padding. Dataset mask is True for Valid, False for Padding.
    # sns.heatmap hides values where mask is True.
    # So we pass ~mask_viz (True for Padding) to hide it.
    sns.heatmap(data_viz, cmap='viridis', cbar=True, annot=True, fmt='.2f', vmin=-1, vmax=max_len + 1, mask=~mask_viz)
    plt.title(f'Data Vector (Valid Length: {valid_len})')
    plt.xlabel('Feature Dimension (1)')
    plt.ylabel('Sequence Length')
    
    # Add a red line to show where padding starts
    plt.axhline(y=valid_len, color='r', linestyle='--', linewidth=2)
    
    # Plot the mask
    plt.subplot(1, 2, 2)
    # Reshape mask to be (max_length, 1) for heatmap visualization
    # Use custom colormap: 0 (Pad) -> Red, 1 (Valid) -> Blue
    from matplotlib.colors import ListedColormap
    cmap_mask = ListedColormap(['#FFB6C1', '#ADD8E6']) # Light Red (Pad) and Light Blue (Valid)
    sns.heatmap(mask_np[:, None], cmap=cmap_mask, cbar=False, annot=True, vmin=0, vmax=1)

    plt.title('Mask (Blue=Valid, Red=Pad)')
    plt.ylabel('Sequence Length')
    plt.xticks([])
    
    plt.tight_layout()
    plt.savefig(out_file_name)
    plt.close()

if __name__ == "__main__":
    plot_sample_from_dataset()

