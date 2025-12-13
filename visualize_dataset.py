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

def plot_sample(data, mask, out_file_name):
    # Convert to numpy for plotting
    data_np = data.numpy()
    mask_np = mask.numpy()
    
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
    
    # Plot the data matrix
    plt.subplot(1, 2, 1)
    sns.heatmap(data_viz, cmap='viridis', cbar=True, annot=True, fmt='.2f')
    plt.title(f'Data Vector (Valid Length: {valid_len})')
    plt.xlabel('Feature Dimension (1)')
    plt.ylabel('Sequence Length')
    
    # Add a red line to show where padding starts
    plt.axhline(y=valid_len, color='r', linestyle='--', linewidth=2)
    
    # Plot the mask
    plt.subplot(1, 2, 2)
    # Reshape mask to be (max_length, 1) for heatmap visualization
    sns.heatmap(mask_np[:, None], cmap='coolwarm', cbar=False, annot=True)

    plt.title('Mask (False=Valid, True=Pad)')
    plt.ylabel('Sequence Length')
    plt.xticks([])
    
    plt.tight_layout()
    plt.savefig(out_file_name)
    print(f"Plot saved to {out_file_name}")

if __name__ == "__main__":
    plot_sample_from_dataset()

