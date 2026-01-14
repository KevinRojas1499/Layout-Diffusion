import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.datasets import get_dataset
from utils.tokenizer import VocabTokenizer

def plot_sample(data, yt, mask, out_file_name, character_tokenizer : VocabTokenizer, max_category=100):
    # Convert to numpy for plotting
    data_np = data.numpy()
    yt_np = yt.numpy() if isinstance(yt, torch.Tensor) else np.array(yt)
    mask_np = mask.numpy()
    
    # Detokenize: convert token IDs to atom names
    yt_detokenized = [character_tokenizer.idx_to_atom[y.item()] for y in yt_np]
    yt_detokenized = np.array(yt_detokenized, dtype=object)  # Convert to numpy array of strings
    
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
    
    # Plot 2: Categorical values (yt) - using detokenized atom names
    plt.subplot(1, 3, 2)
    # Reshape yt_detokenized for heatmap visualization
    yt_viz = yt_detokenized[:, None] if yt_detokenized.ndim == 1 else yt_detokenized
    # Mask out padding positions
    yt_mask_viz = ~mask_np[:, None] if yt_viz.ndim == 2 else ~mask_np
    if yt_viz.ndim == 1:
        yt_mask_viz = yt_mask_viz[:, None]
    
    # For string annotations, we need to create a numeric array for coloring and annotate with strings
    # Create a mapping from unique atoms to numeric codes for coloring
    unique_atoms = np.unique(yt_detokenized[mask_np])  # Only consider valid (non-padded) atoms
    atom_to_code = {atom: idx for idx, atom in enumerate(unique_atoms)}
    yt_codes = np.array([atom_to_code.get(atom, -1) for atom in yt_detokenized])
    yt_codes_viz = yt_codes[:, None] if yt_codes.ndim == 1 else yt_codes
    
    # Use the numeric codes for coloring, but annotate with the actual atom names
    sns.heatmap(yt_codes_viz, cmap='Set3', cbar=True, annot=yt_viz, fmt='', 
                mask=yt_mask_viz, cbar_kws={'label': 'Atom Type'})
    plt.title('Categorical Values (yt) - Atom Names')
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

def plot_molecule(symbols, positions, out_file_name):
    """
    Plots a molecule in 3D based on its QM9-style dictionary.
    
    Args:
        molecule_data (dict): Dictionary containing 'atomic_symbols', 'pos', etc.
    """
    # CPK coloring convention (approximate)
    colors = {
        'H': 'white',
        'C': 'grey',
        'N': 'blue',
        'O': 'red',
        'F': 'green',
        'S': 'yellow',
        'Cl': 'green'
    }
    
    # Atom sizes (approximate relative scales)
    sizes = {
        'H': 100,
        'C': 300,
        'N': 300,
        'O': 300,
        'F': 300,
        'S': 400,
        'Cl': 400
    }
    
    atom_colors = [colors.get(s, 'pink') for s in symbols]
    atom_sizes = [sizes.get(s, 200) for s in symbols]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot atoms
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
               s=atom_sizes, c=atom_colors, edgecolor='black', alpha=1.0)
    
    # Infer and plot bonds based on distance
    # Covalent radii (Angstroms)
    radii = {
        'H': 0.31,
        'C': 0.76,
        'N': 0.71,
        'O': 0.66,
        'F': 0.57,
        'S': 1.05,
        'Cl': 1.02
    }
    
    num_atoms = len(symbols)
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            dist = np.linalg.norm(positions[i] - positions[j])
            
            # Simple bond threshold: sum of radii + tolerance
            threshold = radii.get(symbols[i], 0.7) + radii.get(symbols[j], 0.7) + 0.3
            
            if dist < threshold:
                ax.plot([positions[i, 0], positions[j, 0]],
                        [positions[i, 1], positions[j, 1]],
                        [positions[i, 2], positions[j, 2]],
                        color='black', linewidth=2)
    
    # Label atoms
    for i, sym in enumerate(symbols):
        ax.text(positions[i, 0], positions[i, 1], positions[i, 2], sym, 
                fontsize=10, ha='center', va='center', zorder=10)

    # title = molecule_data.get('canonical_smiles', molecule_data.get('smiles', 'Molecule'))
    # ax.set_title(f"Molecule Structure: {title}")
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    
    # Set equal aspect ratio for 3D plot to prevent distortion
    # Matplotlib 3D doesn't have "axis equal" so we fake it by setting limits
    max_range = np.array([positions[:, 0].max()-positions[:, 0].min(), 
                          positions[:, 1].max()-positions[:, 1].min(), 
                          positions[:, 2].max()-positions[:, 2].min()]).max() / 2.0

    mid_x = (positions[:, 0].max()+positions[:, 0].min()) * 0.5
    mid_y = (positions[:, 1].max()+positions[:, 1].min()) * 0.5
    mid_z = (positions[:, 2].max()+positions[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.savefig(out_file_name)
    plt.close()
