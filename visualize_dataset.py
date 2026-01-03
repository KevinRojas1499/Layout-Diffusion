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
    # Create dummy yt if not present (for euclidean dataset)
    yt = sample.get("label", torch.zeros_like(mask, dtype=torch.long))
    plot_sample(data, yt, mask, 'dataset_visualization.png')

def plot_sample(data, yt, mask, out_file_name, max_category=100):
    # Convert to numpy for plotting
    data_np = data.numpy()
    yt_np = yt.numpy() if isinstance(yt, torch.Tensor) else np.array(yt)
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
    
    # Plotting - now with 3 subplots
    plt.figure(figsize=(15, 6))
    
    max_len = data_np.shape[0]
    
    # Plot 1: Euclidean data
    plt.subplot(1, 3, 1)
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
    
    # Plot 2: Categorical values (yt)
    plt.subplot(1, 3, 2)
    # Reshape yt for heatmap visualization
    yt_viz = yt_np[:, None] if yt_np.ndim == 1 else yt_np
    # Mask out padding positions
    yt_mask_viz = ~mask_np[:, None] if yt_viz.ndim == 2 else ~mask_np
    if yt_viz.ndim == 1:
        yt_mask_viz = yt_mask_viz[:, None]
    
    # Use integer formatting for categorical values
    # Use fixed range for consistent colors across different plots
    vmin_cat = 0
    vmax_cat = max_category
    
    sns.heatmap(yt_viz, cmap='Set3', cbar=True, annot=True, fmt='d', 
                vmin=vmin_cat, vmax=vmax_cat, mask=yt_mask_viz, 
                cbar_kws={'label': 'Category'})
    plt.title('Categorical Values (yt)')
    plt.xlabel('Feature Dimension (1)')
    plt.ylabel('Sequence Length')
    plt.axhline(y=valid_len, color='r', linestyle='--', linewidth=2)
    
    # Plot 3: Mask
    plt.subplot(1, 3, 3)
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

