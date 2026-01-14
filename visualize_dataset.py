from __future__ import annotations
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.datasets import get_dataset
from utils.tokenizer import VocabTokenizer
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
RDKIT_AVAILABLE = True

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
    Plots a molecule in 3D with accurate bond detection.
    
    Args:
        symbols: List of atomic symbols (e.g., ['C', 'H', 'O', ...])
        positions: numpy array of shape (N, 3) with atomic positions in Angstroms
        out_file_name: Output file path for the plot
        smiles: Optional SMILES string for accurate bond detection via RDKit
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
    
    # Get bonds - use RDKit if available and SMILES provided, otherwise use improved distance-based method
    bonds = _get_bonds(symbols, positions, smiles)
    
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


def _get_bonds(symbols, positions, smiles=None):
    """
    Get bonds between atoms using robust distance-based detection.
    Works without SMILES by using accurate bond length thresholds and graph-based validation.
    
    Returns:
        List of tuples (i, j) representing bonds between atoms i and j
    """
    num_atoms = len(symbols)
    
    # Try RDKit-based bond detection if SMILES is available (optional, for validation)
    if smiles and RDKIT_AVAILABLE:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mol = Chem.AddHs(mol)
                rdkit_bonds = []
                for bond in mol.GetBonds():
                    i = bond.GetBeginAtomIdx()
                    j = bond.GetEndAtomIdx()
                    if i < num_atoms and j < num_atoms:
                        rdkit_bonds.append((i, j))
                if len(rdkit_bonds) > 0:
                    return rdkit_bonds
        except Exception:
            pass
    
    # Robust distance-based bond detection (works without SMILES)
    # Bond length thresholds in Angstroms - using maximum reasonable bond lengths
    # These are conservative to avoid false positives
    bond_thresholds = {
        # Single bonds
        ('H', 'H'): 0.80,
        ('H', 'C'): 1.15,
        ('H', 'N'): 1.08,
        ('H', 'O'): 1.02,
        ('H', 'F'): 0.98,
        ('H', 'S'): 1.35,
        ('H', 'Cl'): 1.30,
        ('C', 'C'): 1.70,  # Conservative for single bonds, will catch double/triple too
        ('C', 'N'): 1.60,
        ('C', 'O'): 1.55,
        ('C', 'F'): 1.45,
        ('C', 'S'): 1.90,
        ('C', 'Cl'): 1.85,
        ('N', 'N'): 1.55,
        ('N', 'O'): 1.50,
        ('N', 'F'): 1.45,
        ('O', 'O'): 1.60,
        ('O', 'F'): 1.50,
        ('F', 'F'): 1.50,
        ('S', 'S'): 2.20,
        ('Cl', 'Cl'): 2.20,
    }
    
    # Covalent radii for fallback calculation (in Angstroms)
    covalent_radii = {
        'H': 0.31,
        'C': 0.76,
        'N': 0.71,
        'O': 0.66,
        'F': 0.57,
        'S': 1.05,
        'Cl': 1.02,
        'P': 1.07,
        'Br': 1.20,
    }
    
    # Calculate all pairwise distances
    distances = np.zeros((num_atoms, num_atoms))
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            dist = np.linalg.norm(positions[i] - positions[j])
            distances[i, j] = dist
            distances[j, i] = dist
    
    # Find potential bonds
    potential_bonds = []
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            atom1, atom2 = symbols[i], symbols[j]
            dist = distances[i, j]
            
            # Get threshold for this atom pair
            bond_key = tuple(sorted([atom1, atom2]))
            if bond_key in bond_thresholds:
                threshold = bond_thresholds[bond_key]
            else:
                # Fallback: use sum of covalent radii + 30% tolerance
                r1 = covalent_radii.get(atom1, 0.7)
                r2 = covalent_radii.get(atom2, 0.7)
                threshold = (r1 + r2) * 1.3
            
            # Special cases
            if atom1 == 'H' and atom2 == 'H':
                # H-H bonds are very rare, only if extremely close
                if dist > 0.85:
                    continue
            elif atom1 == 'H' or atom2 == 'H':
                # H can only have one bond typically, but we'll let the graph validation handle this
                pass
            
            if dist < threshold:
                potential_bonds.append((i, j, dist))
    
    # Sort by distance (shorter bonds are more likely to be real)
    potential_bonds.sort(key=lambda x: x[2])
    
    # Graph-based validation: ensure each atom has reasonable connectivity
    # Build adjacency list
    bonds = []
    atom_degrees = {i: 0 for i in range(num_atoms)}
    
    # Maximum expected valency (number of bonds) for each element
    max_valency = {
        'H': 1,
        'C': 4,
        'N': 3,  # Can be 4 with charge, but 3 is typical
        'O': 2,
        'F': 1,
        'S': 6,  # Can form multiple bonds
        'Cl': 1,
        'P': 5,
        'Br': 1,
    }
    
    for i, j, dist in potential_bonds:
        atom1, atom2 = symbols[i], symbols[j]
        max_deg1 = max_valency.get(atom1, 4)
        max_deg2 = max_valency.get(atom2, 4)
        
        # Check if adding this bond would exceed valency
        if atom_degrees[i] < max_deg1 and atom_degrees[j] < max_deg2:
            bonds.append((i, j))
            atom_degrees[i] += 1
            atom_degrees[j] += 1
    
    # Additional pass: for atoms that seem under-connected, look for slightly longer bonds
    # This helps catch cases where the initial threshold was too conservative
    for i in range(num_atoms):
        atom = symbols[i]
        expected_min_bonds = 1 if atom != 'H' else 1
        
        # If atom has no bonds and should have at least one, be more lenient
        if atom_degrees[i] == 0:
            for j in range(num_atoms):
                if i == j:
                    continue
                if (i, j) in bonds or (j, i) in bonds:
                    continue
                
                atom2 = symbols[j]
                dist = distances[i, j]
                bond_key = tuple(sorted([atom, atom2]))
                
                # Use more lenient threshold
                if bond_key in bond_thresholds:
                    threshold = bond_thresholds[bond_key] * 1.15  # 15% more lenient
                else:
                    r1 = covalent_radii.get(atom, 0.7)
                    r2 = covalent_radii.get(atom2, 0.7)
                    threshold = (r1 + r2) * 1.5
                
                max_deg1 = max_valency.get(atom, 4)
                max_deg2 = max_valency.get(atom2, 4)
                
                if dist < threshold and atom_degrees[i] < max_deg1 and atom_degrees[j] < max_deg2:
                    bonds.append((i, j))
                    atom_degrees[i] += 1
                    atom_degrees[j] += 1
                    break  # Only add one bond per iteration
    
    return bonds