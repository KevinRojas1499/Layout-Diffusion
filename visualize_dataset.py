from __future__ import annotations
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.datasets import get_dataset
from utils.tokenizer import VocabTokenizer
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from openbabel import openbabel as ob
import sys
import os
from contextlib import contextmanager

# Suppress OpenBabel error messages
@contextmanager
def suppress_stderr():
    """Context manager to suppress stderr output from OpenBabel (including C++ output)."""
    # Redirect at file descriptor level to catch C++ output
    with open(os.devnull, 'w') as devnull:
        old_stderr_fd = sys.stderr.fileno()
        # Save original stderr
        saved_stderr = os.dup(old_stderr_fd)
        try:
            # Redirect stderr to devnull
            os.dup2(devnull.fileno(), old_stderr_fd)
            yield
        finally:
            # Restore original stderr
            os.dup2(saved_stderr, old_stderr_fd)
            os.close(saved_stderr)

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

def plot_sample_2(data, yt, x_mask, y_mask, out_file_name, character_tokenizer : VocabTokenizer, max_category=100):
    # Convert to numpy for plotting
    data_np = data.detach().cpu().numpy() if isinstance(data, torch.Tensor) else np.array(data)
    yt_np = yt.detach().cpu().numpy() if isinstance(yt, torch.Tensor) else np.array(yt)
    x_mask_np = x_mask.detach().cpu().numpy() if isinstance(x_mask, torch.Tensor) else np.array(x_mask)
    y_mask_np = y_mask.detach().cpu().numpy() if isinstance(y_mask, torch.Tensor) else np.array(y_mask)

    # Detokenize: convert token IDs to symbol names
    yt_detokenized = np.array([character_tokenizer.idx_to_atom[int(token)] for token in yt_np], dtype=object)

    # Reshape 1D data for heatmap visualization if necessary
    if data_np.ndim == 1:
        data_viz = data_np[:, None]
        x_mask_viz = x_mask_np[:, None]
    else:
        data_viz = data_np
        x_mask_viz = x_mask_np
        if x_mask_viz.ndim == 1:
            x_mask_viz = np.tile(x_mask_viz[:, None], (1, data_viz.shape[1]))

    # Modalities can have different valid lengths
    valid_len_x = int(x_mask_np.sum())
    valid_len_y = int(y_mask_np.sum())

    # Plotting - similar to plot_sample but with one mask per modality
    plt.figure(figsize=(18, 6))
    max_len = data_np.shape[0]

    # Plot 1: Euclidean data
    plt.subplot(1, 4, 1)
    sns.heatmap(
        data_viz,
        cmap='viridis',
        cbar=True,
        annot=True,
        fmt='.2f',
        vmin=-1,
        vmax=max_len + 1,
        mask=~x_mask_viz,
    )
    plt.title(f'Data Vector (Valid Length: {valid_len_x})')
    plt.xlabel('Feature Dimension (1)')
    plt.ylabel('Sequence Length')
    plt.axhline(y=valid_len_x, color='r', linestyle='--', linewidth=2)

    # Plot 2: Categorical values (yt)
    plt.subplot(1, 4, 2)
    yt_viz = yt_detokenized[:, None] if yt_detokenized.ndim == 1 else yt_detokenized
    yt_mask_viz = ~y_mask_np[:, None] if yt_viz.ndim == 2 else ~y_mask_np
    if yt_viz.ndim == 1:
        yt_mask_viz = yt_mask_viz[:, None]

    valid_atoms = yt_detokenized[y_mask_np]
    unique_atoms = np.unique(valid_atoms) if len(valid_atoms) > 0 else np.array(['<pad>'], dtype=object)
    atom_to_code = {atom: idx for idx, atom in enumerate(unique_atoms)}
    yt_codes = np.array([atom_to_code.get(atom, -1) for atom in yt_detokenized])
    yt_codes_viz = yt_codes[:, None] if yt_codes.ndim == 1 else yt_codes

    sns.heatmap(
        yt_codes_viz,
        cmap='Set3',
        cbar=True,
        annot=yt_viz,
        fmt='',
        mask=yt_mask_viz,
        cbar_kws={'label': 'Token Type'},
    )
    plt.title(f'Categorical Values (Valid Length: {valid_len_y})')
    plt.xlabel('Feature Dimension (1)')
    plt.ylabel('Sequence Length')
    plt.axhline(y=valid_len_y, color='r', linestyle='--', linewidth=2)

    # Plot 3: x mask
    plt.subplot(1, 4, 3)
    from matplotlib.colors import ListedColormap
    cmap_mask = ListedColormap(['#FFB6C1', '#ADD8E6'])  # Pad, Valid
    sns.heatmap(x_mask_np[:, None], cmap=cmap_mask, cbar=False, annot=True, vmin=0, vmax=1)
    plt.title('x_mask (Blue=Valid, Red=Pad)')
    plt.ylabel('Sequence Length')
    plt.xticks([])

    # Plot 4: y mask
    plt.subplot(1, 4, 4)
    sns.heatmap(y_mask_np[:, None], cmap=cmap_mask, cbar=False, annot=True, vmin=0, vmax=1)
    plt.title('y_mask (Blue=Valid, Red=Pad)')
    plt.ylabel('Sequence Length')
    plt.xticks([])

    plt.tight_layout()
    plt.savefig(out_file_name)
    plt.close()

def plot_molecule(symbols, positions, out_file_name, smiles=None):
    """
    Plots a molecule in 3D with accurate bond detection using OpenBabel pipeline.
    Matches evaluation methodology: xyz → sdf (OpenBabel) → RDKit.
    
    Args:
        symbols: List of atomic symbols (e.g., ['C', 'H', 'O', ...])
        positions: numpy array of shape (N, 3) with atomic positions in Angstroms
        out_file_name: Output file path for the plot
        smiles: Optional SMILES string (not used, kept for backward compatibility)
    """
    # Convert to numpy if needed
    if isinstance(positions, torch.Tensor):
        positions = positions.cpu().numpy()
    positions = np.array(positions)
    symbols = list(symbols)
    
    # Filter out padding (atoms at origin with zero coordinates)
    valid_mask = np.any(positions != 0, axis=1) | (np.sum(np.abs(positions), axis=1) > 1e-6)
    if valid_mask.sum() < len(symbols):
        symbols = [s for i, s in enumerate(symbols) if valid_mask[i]]
        positions = positions[valid_mask]
    
    # CPK coloring convention
    colors = {
        'H': 'white',
        'C': 'grey',
        'N': 'blue',
        'O': 'red',
        'F': 'green',
        'S': 'yellow',
        'Cl': 'green',
        'P': 'orange',
        'Br': 'darkred'
    }
    
    # Atom sizes (approximate relative scales)
    sizes = {
        'H': 100,
        'C': 300,
        'N': 300,
        'O': 300,
        'F': 300,
        'S': 400,
        'Cl': 400,
        'P': 400,
        'Br': 400
    }
    
    atom_colors = [colors.get(s, 'pink') for s in symbols]
    atom_sizes = [sizes.get(s, 200) for s in symbols]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot atoms
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
               s=atom_sizes, c=atom_colors, edgecolor='black', alpha=1.0, linewidths=1.5)
    
    # Get bonds using OpenBabel pipeline (matches evaluation methodology)
    bonds = _get_bonds_via_openbabel(symbols, positions)
    
    # Plot bonds
    for i, j in bonds:
        ax.plot([positions[i, 0], positions[j, 0]],
                [positions[i, 1], positions[j, 1]],
                [positions[i, 2], positions[j, 2]],
                color='black', linewidth=2, zorder=1)
    
    # Label atoms
    for i, sym in enumerate(symbols):
        ax.text(positions[i, 0], positions[i, 1], positions[i, 2], sym, 
                fontsize=10, ha='center', va='center', zorder=10, 
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))

    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    
    # Use fixed axis limits for consistent visualization
    axis_limits = get_fixed_axis_limits()
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = axis_limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    
    plt.tight_layout()
    plt.savefig(out_file_name, dpi=150, bbox_inches='tight')
    plt.close()


def get_fixed_axis_limits():
    """
    Get fixed axis limits for consistent visualization across all molecules.
    These limits are independent of the molecule size and ensure all plots use the same scale.
    
    Returns:
        Tuple of ((x_min, x_max), (y_min, y_max), (z_min, z_max))
    """
    # Fixed limits suitable for QM9 molecules (small organic molecules)
    # Range of -15 to 15 Angstroms should accommodate most molecules with some padding
    limit = 3.0
    return ((-limit, limit), (-limit, limit), (-limit, limit))


def compute_axis_limits(positions, padding=2.0):
    """
    Compute fixed axis limits for animation consistency.
    
    Args:
        positions: numpy array of shape (N, 3) with atomic positions
        padding: Additional padding around the molecule in Angstroms
    
    Returns:
        Tuple of ((x_min, x_max), (y_min, y_max), (z_min, z_max))
    """
    if len(positions) == 0:
        return get_fixed_axis_limits()
    
    # Filter out invalid positions (at origin)
    valid_mask = np.any(positions != 0, axis=1) | (np.sum(np.abs(positions), axis=1) > 1e-6)
    if valid_mask.sum() == 0:
        return get_fixed_axis_limits()
    
    valid_positions = positions[valid_mask]
    
    max_range = np.array([
        valid_positions[:, 0].max() - valid_positions[:, 0].min(),
        valid_positions[:, 1].max() - valid_positions[:, 1].min(),
        valid_positions[:, 2].max() - valid_positions[:, 2].min()
    ]).max() / 2.0
    
    if max_range == 0:
        max_range = 5.0
    
    mid_x = (valid_positions[:, 0].max() + valid_positions[:, 0].min()) * 0.5
    mid_y = (valid_positions[:, 1].max() + valid_positions[:, 1].min()) * 0.5
    mid_z = (valid_positions[:, 2].max() + valid_positions[:, 2].min()) * 0.5
    
    return (
        (mid_x - max_range - padding, mid_x + max_range + padding),
        (mid_y - max_range - padding, mid_y + max_range + padding),
        (mid_z - max_range - padding, mid_z + max_range + padding)
    )


def plot_molecule_with_mask(symbols, positions, mask, out_file_name, character_tokenizer, t=None, axis_limits=None):
    """
    Plots a molecule in 3D showing masked tokens with different visualization.
    Masked tokens are shown as transparent/hollow atoms.
    
    Args:
        symbols: List of atomic symbols (e.g., ['C', 'H', 'O', ...]) or token IDs (Tensor/array)
        positions: numpy array of shape (N, 3) with atomic positions in Angstroms
        mask: numpy array or tensor of shape (N,) indicating which tokens are masked (True=masked)
        out_file_name: Output file path for the plot
        character_tokenizer: Tokenizer to decode symbols if they are token IDs
        t: Optional timestep value to display
        axis_limits: Optional tuple ((x_min, x_max), (y_min, y_max), (z_min, z_max)) to fix axis limits
    """
    # Convert to numpy if needed
    if isinstance(positions, torch.Tensor):
        positions = positions.cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    positions = np.array(positions)
    mask = np.array(mask, dtype=bool)
    
    # Get special token IDs before converting
    bos_token_id = character_tokenizer.bos_token_id
    eos_token_id = character_tokenizer.eos_token_id
    pad_token_id = character_tokenizer.pad_token_id
    mask_token_id = character_tokenizer.mask_token_id
    
    # Store original token IDs before conversion
    if isinstance(symbols, torch.Tensor):
        token_ids = symbols.cpu().numpy()
    elif isinstance(symbols, (list, np.ndarray)) and len(symbols) > 0:
        first_elem = symbols[0] if isinstance(symbols, list) else symbols[0].item() if isinstance(symbols, np.ndarray) else symbols[0]
        if isinstance(first_elem, (int, np.integer)):
            token_ids = np.array([int(token) for token in symbols])
        else:
            token_ids = None
    else:
        token_ids = None
    
    # Identify special tokens from token IDs
    if token_ids is not None:
        is_bos = (token_ids == bos_token_id)
        is_eos = (token_ids == eos_token_id)
        is_pad = (token_ids == pad_token_id)
    else:
        # Already converted to strings, check by string comparison
        is_bos = np.array([s == '<BOS>' or s == character_tokenizer.idx_to_atom.get(bos_token_id, '') for s in symbols])
        is_eos = np.array([s == '<EOS>' or s == character_tokenizer.idx_to_atom.get(eos_token_id, '') for s in symbols])
        is_pad = np.array([s == '<pad>' or s == character_tokenizer.idx_to_atom.get(pad_token_id, '') for s in symbols])
    
    # Decode symbols if they are token IDs
    if isinstance(symbols, torch.Tensor):
        symbols_tensor = symbols.cpu()
        # Convert token IDs to symbol strings
        symbols = [character_tokenizer.idx_to_atom[token.item()] for token in symbols_tensor]
    elif isinstance(symbols, (list, np.ndarray)) and len(symbols) > 0:
        # Check if first element is an integer (token ID)
        first_elem = symbols[0] if isinstance(symbols, list) else symbols[0].item() if isinstance(symbols, np.ndarray) else symbols[0]
        if isinstance(first_elem, (int, np.integer)):
            # Convert token IDs to symbol strings
            symbols = [character_tokenizer.idx_to_atom[int(token)] for token in symbols]
    else:
        symbols = list(symbols)
    
    # Filter out padding (atoms at origin with zero coordinates)
    # Use a very strict threshold - only consider positions at exactly/nearly origin as invalid
    # This ensures we preserve actual positions for masked tokens that are being denoised
    position_magnitude = np.sum(np.abs(positions), axis=1)
    # Very strict: only positions that are essentially at origin (within 1e-5) are considered invalid
    # This way, masked tokens with any meaningful position will be shown at their actual location
    valid_positions = position_magnitude > 1e-5  # Very strict threshold to preserve actual positions
    
    # Create display mask: exclude BOS, EOS, and padding
    # But we want to show masked tokens even if they don't have valid positions yet
    display_mask = ~is_pad & ~is_bos & ~is_eos
    
    if display_mask.sum() == 0:
        # Nothing to plot
        return
    
    # Apply display mask
    symbols = [symbols[i] for i in range(len(symbols)) if display_mask[i]]
    positions = positions[display_mask]
    mask = mask[display_mask] if len(mask) == len(display_mask) else mask[:len(display_mask)][display_mask]
    valid_positions = valid_positions[display_mask]
    
    # IMPORTANT: Masked tokens are set to origin (0,0,0) in the interpolant
    # We should show them at origin, but spread them slightly so they're visible
    # Check which masked tokens are at origin
    masked_at_origin = [i for i in range(len(positions)) if mask[i] and not valid_positions[i]]
    
    # For masked tokens at origin, keep them at origin but spread them minimally for visibility
    # Masked tokens are set to (0,0,0) in the interpolant, so we show them clustered at origin
    if len(masked_at_origin) > 0:
        # Spread masked tokens in a very small sphere around origin (0.3 Å radius)
        # This makes them visible while clearly showing they're at origin
        spread_radius = 0.3  # Very small spread - just enough to see multiple tokens
        for idx, i in enumerate(masked_at_origin):
            if len(masked_at_origin) == 1:
                # Single masked token - keep it exactly at origin
                positions[i] = np.array([0.0, 0.0, 0.0])
            else:
                # Multiple masked tokens - spread them in a tiny sphere around origin
                angle = 2 * np.pi * idx / len(masked_at_origin)
                z_offset = 0.1 * np.sin(2 * angle)  # Very small z variation
                positions[i] = spread_radius * np.array([
                    np.cos(angle),
                    np.sin(angle),
                    z_offset
                ])
    
    # Note: Masked tokens with valid positions (not at origin) will be shown at their actual positions
    # This can happen during denoising when positions are being updated
    
    # Count masked tokens
    num_masked = mask.sum()
    num_unmasked = (~mask).sum()
    
    # CPK coloring convention
    colors = {
        'H': 'white',
        'C': 'grey',
        'N': 'blue',
        'O': 'red',
        'F': 'green',
        'S': 'yellow',
        'Cl': 'green',
        'P': 'orange',
        'Br': 'darkred',
        '<M>': 'purple',  # Mask token
        '<pad>': 'lightgray',  # Padding
    }
    
    # Atom sizes (approximate relative scales)
    sizes = {
        'H': 100,
        'C': 300,
        'N': 300,
        'O': 300,
        'F': 300,
        'S': 400,
        'Cl': 400,
        'P': 400,
        'Br': 400,
        '<M>': 250,
        '<pad>': 150,
    }
    
    # Separate masked and unmasked atoms
    unmasked_indices = np.where(~mask)[0]
    masked_indices = np.where(mask)[0]
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot unmasked atoms (normal, opaque)
    if len(unmasked_indices) > 0:
        unmasked_symbols = [symbols[i] for i in unmasked_indices]
        unmasked_positions = positions[unmasked_indices]
        unmasked_colors = [colors.get(s, 'pink') for s in unmasked_symbols]
        unmasked_sizes = [sizes.get(s, 200) for s in unmasked_symbols]
        
        ax.scatter(unmasked_positions[:, 0], unmasked_positions[:, 1], unmasked_positions[:, 2], 
                   s=unmasked_sizes, c=unmasked_colors, edgecolor='black', alpha=1.0, linewidths=1.5,
                   label='Unmasked', zorder=5)
    
    # Plot masked atoms (transparent/hollow with red edge) at their ACTUAL positions from trajectory
    # Note: positions[masked_indices] contains the actual positions from the trajectory data
    # We only modified positions for masked tokens that were at origin (no valid position)
    # All other masked tokens are shown at their real positions from the sampling process
    if len(masked_indices) > 0:
        masked_symbols = [symbols[i] for i in masked_indices]
        masked_positions = positions[masked_indices]  # Actual positions from trajectory (or default if was at origin)
        masked_colors = [colors.get(s, 'purple') for s in masked_symbols]
        masked_sizes = [sizes.get(s, 200) for s in masked_symbols]
        
        # Plot as hollow/transparent with red edge
        # This shows masked tokens at their actual positions from the trajectory
        # If a masked token had a valid position, it's shown there; if not, it's at the default location
        ax.scatter(masked_positions[:, 0], masked_positions[:, 1], masked_positions[:, 2], 
                   s=masked_sizes, c=masked_colors, edgecolor='red', alpha=0.3, linewidths=2.5,
                   label='Masked', zorder=4)
        # Add a second layer with just edges for visibility
        ax.scatter(masked_positions[:, 0], masked_positions[:, 1], masked_positions[:, 2], 
                   s=masked_sizes, facecolors='none', edgecolor='red', alpha=0.8, linewidths=2.5,
                   zorder=6)
    
    # Get bonds using OpenBabel pipeline (only for unmasked atoms)
    if len(unmasked_indices) > 0:
        unmasked_symbols_for_bonds = [symbols[i] for i in unmasked_indices]
        unmasked_positions_for_bonds = positions[unmasked_indices]
        bonds = _get_bonds_via_openbabel(unmasked_symbols_for_bonds, unmasked_positions_for_bonds)
        
        # Map bond indices back to original positions
        idx_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(unmasked_indices)}
        for i, j in bonds:
            if i < len(unmasked_indices) and j < len(unmasked_indices):
                orig_i = unmasked_indices[i]
                orig_j = unmasked_indices[j]
                ax.plot([positions[orig_i, 0], positions[orig_j, 0]],
                        [positions[orig_i, 1], positions[orig_j, 1]],
                        [positions[orig_i, 2], positions[orig_j, 2]],
                        color='black', linewidth=2, zorder=1, alpha=0.6)
    
    # Label atoms
    for i, sym in enumerate(symbols):
        if mask[i]:
            # Masked atoms: show as "M" or the symbol with a red background
            label = f'M' if sym == '<M>' else sym
            ax.text(positions[i, 0], positions[i, 1], positions[i, 2], label, 
                    fontsize=9, ha='center', va='center', zorder=10, 
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='red', alpha=0.5, edgecolor='red', linewidth=1.5),
                    color='white', weight='bold')
        else:
            # Unmasked atoms: normal label
            ax.text(positions[i, 0], positions[i, 1], positions[i, 2], sym, 
                    fontsize=10, ha='center', va='center', zorder=10, 
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))
    
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    
    # Create title with mask information
    title_parts = [f'Masked: {num_masked}, Unmasked: {num_unmasked}']
    if t is not None:
        title_parts.append(f't={t:.3f}')
    ax.set_title(' / '.join(title_parts), fontsize=12, pad=20)
    
    # Add legend
    ax.legend(loc='upper left')
    
    # Use fixed axis limits for consistent visualization
    # If axis_limits is provided (for trajectory animation), use those; otherwise use global fixed limits
    if axis_limits is not None:
        # Use provided fixed axis limits for animation consistency
        (x_min, x_max), (y_min, y_max), (z_min, z_max) = axis_limits
    else:
        # Use global fixed axis limits
        axis_limits = get_fixed_axis_limits()
        (x_min, x_max), (y_min, y_max), (z_min, z_max) = axis_limits
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    
    plt.tight_layout()
    plt.savefig(out_file_name, dpi=150, bbox_inches='tight')
    plt.close()


def _get_bonds_via_openbabel(symbols, positions):
    """
    Get bonds between atoms using OpenBabel pipeline (matches evaluation methodology).
    Converts xyz → sdf → RDKit to get accurate bond structure.
    
    Args:
        symbols: List of atomic symbols
        positions: numpy array of shape (N, 3) with atomic positions
    
    Returns:
        List of tuples (i, j) representing bonds between atoms i and j
    """
    try:
        # Convert xyz → sdf using OpenBabel (suppress error messages)
        xyz_lines = [f"{len(symbols)}\n", "Molecule\n"]
        for symbol, pos in zip(symbols, positions):
            xyz_lines.append(f"{symbol:2s} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}\n")
        xyz_str = "".join(xyz_lines)
        
        # Convert xyz to sdf using OpenBabel
        with suppress_stderr():
            conv = ob.OBConversion()
            conv.SetInAndOutFormats("xyz", "sdf")
            
            obmol = ob.OBMol()
            if conv.ReadString(obmol, xyz_str):
                # Convert to sdf string
                sdf_str = conv.WriteString(obmol)
                
                # Read sdf into RDKit
                mol = Chem.MolFromMolBlock(sdf_str, sanitize=False)
                if mol is not None:
                    try:
                        Chem.SanitizeMol(mol)
                        # Extract bonds from RDKit molecule
                        bonds = []
                        for bond in mol.GetBonds():
                            i = bond.GetBeginAtomIdx()
                            j = bond.GetEndAtomIdx()
                            if i < len(symbols) and j < len(symbols):
                                bonds.append((i, j))
                        return bonds
                    except:
                        pass
    except Exception:
        pass
    
    # Fallback: return empty list if conversion fails
    # (molecule will be plotted without bonds rather than with incorrect bonds)
    return []