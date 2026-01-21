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
    
    # Set equal aspect ratio for 3D plot to prevent distortion
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