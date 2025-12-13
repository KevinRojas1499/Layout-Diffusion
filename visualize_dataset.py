import torch
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

def plot_sample(data, mask, cur_set, out_file_name):
    # Convert to numpy for plotting
    data_np = data.numpy()
    mask_np = mask.numpy()
    cur_set = cur_set.numpy()
    
    print(f"Data shape: {data_np.shape}")
    print(f"Mask shape: {mask_np.shape}")

    # Reshape 1D data for heatmap visualization if necessary
    if data_np.ndim == 1:
        data_viz = data_np[:, None]
    else:
        data_viz = data_np
    
    # Calculate valid length
    # Mask is False for valid tokens, True for padding
    valid_len = (~mask_np).sum()
    print(f"Valid length: {valid_len}")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    max_len = data_np.shape[0]
    
    # Plot the ordered set (indices) - simple grayscale
    plt.subplot(1, 3 ,1)
    sns.heatmap(cur_set[:,None], annot=True, cbar=False, cmap="Greys", fmt='.0f')
    plt.title(f'Ordered Set')


    plt.subplot(1, 3, 2)
    # Fixed scale from -1 to max_len + 1
    sns.heatmap(data_viz, cmap='viridis', cbar=True, annot=True, fmt='.2f', vmin=-1, vmax=max_len + 1)
    plt.title(f'Data Vector (Valid Length: {valid_len})')
    plt.xlabel('Feature Dimension (1)')
    plt.ylabel('Sequence Length')
    
    # Add a red line to show where padding starts
    plt.axhline(y=valid_len, color='r', linestyle='--', linewidth=2)
    
    # Plot the mask
    plt.subplot(1, 3, 3)
    # Reshape mask to be (max_length, 1) for heatmap visualization
    # Use custom colormap: 0 (Valid) -> Blue, 1 (Pad) -> Red
    from matplotlib.colors import ListedColormap
    cmap_mask = ListedColormap(['#ADD8E6', '#FFB6C1']) # Light Blue and Light Red for better visibility with text
    sns.heatmap(mask_np[:, None], cmap=cmap_mask, cbar=False, annot=True, vmin=0, vmax=1)

    plt.title('Mask (Blue=Valid, Red=Pad)')
    plt.ylabel('Sequence Length')
    plt.xticks([])
    
    plt.tight_layout()
    plt.savefig(out_file_name)
    plt.close()
    print(f"Plot saved to {out_file_name}")

if __name__ == "__main__":
    plot_sample_from_dataset()

