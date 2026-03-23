"""
Evaluation script for comparing generated vs real molecule distributions using UMAP.
Based on the paper's evaluation methodology using molecular fingerprints.
"""

import torch
import numpy as np
import matplotlib
# Use non-interactive backend to avoid tkinter issues in headless environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import os
import warnings
import sys
from contextlib import contextmanager

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit import RDLogger
import umap
from openbabel import openbabel as ob

# Suppress deprecation warnings for GetMorganFingerprintAsBitVect
warnings.filterwarnings("ignore", message=".*GetMorganFingerprintAsBitVect.*", category=DeprecationWarning)
# Suppress RDKit error/warning messages (they're too verbose for invalid molecules)
RDLogger.DisableLog('rdApp.*')

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


def compute_molecular_fingerprints(symbols_list: List[List[str]], positions_list: List[np.ndarray], 
                                   radius: int = 2, n_bits: int = 2048,
                                   fingerprint_type: str = 'rdkit',
                                   filter_invalid: bool = True,
                                   return_stats: bool = False):
    """
    Compute molecular fingerprints for molecules using OpenBabel pipeline.
    Matches paper methodology: xyz → sdf (OpenBabel) → RDKit.
    
    Args:
        symbols_list: List of lists, where each inner list contains atomic symbols
        positions_list: List of numpy arrays, where each array is (N, 3) atomic positions
        radius: Radius for Morgan fingerprint (default 2)
        n_bits: Number of bits in fingerprint (default 2048)
        fingerprint_type: Type of fingerprint ('morgan' or 'rdkit')
        filter_invalid: If True, filter out invalid and not-fully-connected molecules
    
    Returns:
        Tuple of (fingerprints, valid_indices) or (fingerprints, valid_indices, stats) if return_stats:
        - fingerprints: numpy array of shape (num_molecules, n_bits) with binary fingerprints
        - valid_indices: List of original indices that passed filtering
        - stats: dict with counts for total/valid/invalid/not_connected/failed
    """
    fingerprints = []
    valid_indices = []
    failed_count = 0
    invalid_count = 0
    not_connected_count = 0
    
    for idx, (symbols, positions) in enumerate(zip(symbols_list, positions_list)):
        try:
            # Use OpenBabel pipeline: xyz → sdf → RDKit (matches paper methodology)
            mol, _, is_valid, is_fully_connected = _positions_to_molecule_via_openbabel(symbols, positions)
            
            if mol is None:
                failed_count += 1
                continue
            
            # Apply filtering if requested
            if filter_invalid:
                if not is_valid:
                    invalid_count += 1
                    continue
                if not is_fully_connected:
                    not_connected_count += 1
                    continue
            
            # Compute fingerprint based on type
            if fingerprint_type.lower() == 'morgan':
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
            else:
                fp = Chem.RDKFingerprint(mol, maxPath=7, fpSize=n_bits)
            
            fingerprints.append(np.array(fp))
            valid_indices.append(idx)
        except Exception as e:
            failed_count += 1
            if idx < 5:  # Only print first few errors
                print(f"Warning: Failed to compute fingerprint for molecule {idx}: {e}")
            continue
    
    if len(fingerprints) == 0:
        raise ValueError(f"No valid fingerprints computed (failed for all {len(symbols_list)} molecules)")
    
    # Logging + stats
    total_processed = len(symbols_list)
    total_valid = len(fingerprints)
    stats = {
        'total': total_processed,
        'valid': total_valid,
        'invalid': invalid_count,
        'not_connected': not_connected_count,
        'failed': failed_count
    }
    if filter_invalid:
        print(f"Fingerprint computation summary:")
        print(f"  Total molecules: {total_processed}")
        print(f"  Valid (RDKit valid=true): {total_valid + invalid_count + not_connected_count}")
        print(f"  Invalid (RDKit valid=false): {invalid_count}")
        print(f"  Not fully connected: {not_connected_count}")
        print(f"  Failed conversion: {failed_count}")
        print(f"  Final valid count: {total_valid}")
    else:
        if failed_count > 0:
            print(f"Successfully computed {total_valid} fingerprints ({failed_count} failed)")
    
    if return_stats:
        return np.array(fingerprints), valid_indices, stats
    return np.array(fingerprints), valid_indices


def _xyz_to_sdf_via_openbabel(symbols: List[str], positions: np.ndarray) -> Optional[str]:
    """
    Convert atomic symbols and positions to SDF format using OpenBabel.
    This matches the paper's methodology: xyz → sdf conversion via OpenBabel.
    
    Args:
        symbols: List of atomic symbols
        positions: numpy array of shape (N, 3) with atomic positions in Angstroms
    
    Returns:
        SDF string or None if conversion fails
    """
    try:
        # Create xyz string for input
        xyz_lines = [f"{len(symbols)}\n", "Generated molecule\n"]
        for symbol, pos in zip(symbols, positions):
            xyz_lines.append(f"{symbol:2s} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}\n")
        xyz_str = "".join(xyz_lines)
        
        # Convert xyz to sdf using OpenBabel (suppress error messages)
        with suppress_stderr():
            conv = ob.OBConversion()
            conv.SetInAndOutFormats("xyz", "sdf")
            
            obmol = ob.OBMol()
            # Read xyz string into molecule
            if conv.ReadString(obmol, xyz_str):
                # Convert to sdf string
                sdf_str = conv.WriteString(obmol)
                return sdf_str
            else:
                return None
    except Exception as e:
        return None


def _is_fully_connected(mol) -> bool:
    """
    Check if molecule is fully connected (all atoms in a single connected component).
    
    Args:
        mol: RDKit molecule object
    
    Returns:
        True if fully connected, False otherwise
    """
    if mol is None:
        return False
    
    try:
        # Get connected components
        num_components = len(Chem.GetMolFrags(mol))
        return num_components == 1
    except:
        return False


def _sdf_to_rdkit_molecule(sdf_str: str) -> Optional[object]:
    """
    Read SDF string into RDKit molecule object.
    
    Args:
        sdf_str: SDF format string
    
    Returns:
        RDKit molecule object or None if parsing fails
    """
    try:
        # Read from SDF string
        mol = Chem.MolFromMolBlock(sdf_str, sanitize=False)
        if mol is None:
            return None
        
        # Sanitize to check validity
        try:
            Chem.SanitizeMol(mol)
            return mol
        except:
            # Invalid molecule
            return None
    except:
        return None


def _positions_to_molecule_via_openbabel(symbols: List[str], positions: np.ndarray) -> Tuple[Optional[object], Optional[str], bool, bool]:
    """
    Convert atomic symbols and positions to RDKit molecule using OpenBabel pipeline.
    Matches paper methodology: xyz → sdf (OpenBabel) → RDKit.
    
    Args:
        symbols: List of atomic symbols
        positions: numpy array of shape (N, 3) with atomic positions
    
    Returns:
        Tuple of (mol, smiles, is_valid, is_fully_connected):
        - mol: RDKit molecule object or None
        - smiles: SMILES string or None
        - is_valid: True if RDKit valid flag is True
        - is_fully_connected: True if molecule is fully connected
    """
    # Convert xyz → sdf using OpenBabel
    sdf_str = _xyz_to_sdf_via_openbabel(symbols, positions)
    if sdf_str is None:
        return None, None, False, False
    
    # Read sdf into RDKit
    mol = _sdf_to_rdkit_molecule(sdf_str)
    if mol is None:
        return None, None, False, False
    
    # Check validity: if we got here, sanitization succeeded, so molecule is valid
    # (RDKit's sanitization in _sdf_to_rdkit_molecule ensures validity)
    is_valid = True
    
    # Check if fully connected
    is_fully_connected = _is_fully_connected(mol)
    
    # Extract SMILES if possible
    smiles = None
    try:
        smiles = Chem.MolToSmiles(mol)
    except:
        pass
    
    return mol, smiles, is_valid, is_fully_connected


def compute_umap_embedding(fingerprints: np.ndarray, n_components: int = 2, 
                          n_neighbors: Optional[int] = None, min_dist: Optional[float] = None, 
                          random_state: Optional[int] = None) -> np.ndarray:
    """
    Compute UMAP embedding of molecular fingerprints.
    
    Note: UMAP embeddings are deterministic when:
    - random_state is set (as done here)
    - The same fingerprints are provided in the same order
    - The same UMAP parameters are used
    
    However, if different molecules are sampled or fingerprints fail for different
    molecules between runs, the embeddings will differ. Use fixed random seeds
    in extract_molecules_from_qm9_dataset and evaluate_molecule_distributions
    for full reproducibility.
    
    Args:
        fingerprints: numpy array of shape (N, n_bits) with binary fingerprints
        n_components: Number of dimensions for embedding (default 2)
        n_neighbors: Number of neighbors for UMAP (default 15)
        min_dist: Minimum distance in embedding space (default 0.1)
        random_state: Random seed for reproducibility
    
    Returns:
        numpy array of shape (N, n_components) with UMAP embedding
    """
    # Suppress UMAP warnings about n_jobs when random_state is set
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*n_jobs value.*overridden.*')
        umap_kwargs = {
            'n_components': n_components,
            'random_state': random_state,
            'verbose': False
        }
        if n_neighbors is not None:
            umap_kwargs['n_neighbors'] = n_neighbors
        if min_dist is not None:
            umap_kwargs['min_dist'] = min_dist
        reducer = umap.UMAP(**umap_kwargs)
        
        embedding = reducer.fit_transform(fingerprints)
    
    return embedding


def compute_molecular_properties(symbols_list: List[List[str]], positions_list: List[np.ndarray],
                                filter_invalid: bool = True) -> dict:
    """
    Compute molecular properties for each molecule using OpenBabel pipeline.
    Matches paper methodology: xyz → sdf (OpenBabel) → RDKit.
    
    Properties computed:
    - mw: Molecular weight
    - logp: LogP (octanol-water partition coefficient)
    - hbd: Number of hydrogen-bond donors
    - hba: Number of hydrogen-bond acceptors
    
    Args:
        symbols_list: List of lists of atomic symbols
        positions_list: List of numpy arrays of atomic positions
        filter_invalid: If True, filter out invalid and not-fully-connected molecules
    
    Returns:
        Dictionary with keys: 'mw', 'logp', 'hbd', 'hba'
        Each value is a list of property values for each molecule
    """
    properties = {
        'mw': [],
        'logp': [],
        'hbd': [],
        'hba': []
    }
    
    failed_count = 0
    invalid_count = 0
    not_connected_count = 0
    
    for idx, (symbols, positions) in enumerate(zip(symbols_list, positions_list)):
        try:
            # Use OpenBabel pipeline: xyz → sdf → RDKit (matches paper methodology)
            mol, _, is_valid, is_fully_connected = _positions_to_molecule_via_openbabel(symbols, positions)
            
            # Apply filtering if requested
            if filter_invalid:
                if mol is None or not is_valid:
                    invalid_count += 1
                    # Append NaN for filtered molecules
                    properties['mw'].append(np.nan)
                    properties['logp'].append(np.nan)
                    properties['hbd'].append(np.nan)
                    properties['hba'].append(np.nan)
                    continue
                if not is_fully_connected:
                    not_connected_count += 1
                    # Append NaN for filtered molecules
                    properties['mw'].append(np.nan)
                    properties['logp'].append(np.nan)
                    properties['hbd'].append(np.nan)
                    properties['hba'].append(np.nan)
                    continue
            
            if mol is None:
                failed_count += 1
                # Append NaN for failed molecules
                properties['mw'].append(np.nan)
                properties['logp'].append(np.nan)
                properties['hbd'].append(np.nan)
                properties['hba'].append(np.nan)
                continue
            
            # Compute properties
            try:
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                hbd = Descriptors.NumHDonors(mol)
                hba = Descriptors.NumHAcceptors(mol)
                
                properties['mw'].append(mw)
                properties['logp'].append(logp)
                properties['hbd'].append(hbd)
                properties['hba'].append(hba)
            except Exception as e:
                failed_count += 1
                # Append NaN for failed computations
                properties['mw'].append(np.nan)
                properties['logp'].append(np.nan)
                properties['hbd'].append(np.nan)
                properties['hba'].append(np.nan)
                if idx < 5:  # Only print first few errors
                    print(f"Warning: Failed to compute properties for molecule {idx}: {e}")
                continue
                
        except Exception as e:
            failed_count += 1
            properties['mw'].append(np.nan)
            properties['logp'].append(np.nan)
            properties['hbd'].append(np.nan)
            properties['hba'].append(np.nan)
            if idx < 5:
                print(f"Warning: Error processing molecule {idx}: {e}")
            continue
    
    # Logging
    total_processed = len(symbols_list)
    total_valid = sum(1 for mw in properties['mw'] if not np.isnan(mw))
    if filter_invalid:
        print(f"Property computation summary:")
        print(f"  Total molecules: {total_processed}")
        print(f"  Valid (RDKit valid=true): {total_valid + invalid_count + not_connected_count}")
        print(f"  Invalid (RDKit valid=false): {invalid_count}")
        print(f"  Not fully connected: {not_connected_count}")
        print(f"  Failed conversion: {failed_count}")
        print(f"  Final valid count: {total_valid}")
    else:
        if failed_count > 0:
            print(f"Successfully computed properties for {total_valid} molecules ({failed_count} failed)")
    
    return properties


def compute_atom_counts(symbols_list: List[List[str]]) -> dict:
    """
    Compute atom counts for each molecule.
    
    Args:
        symbols_list: List of lists of atomic symbols
    
    Returns:
        Dictionary with keys: 'total', 'C', 'N', 'O', 'F', 'H'
        Each value is a list of counts for each molecule
    """
    counts = {
        'total': [],
        'C': [],
        'N': [],
        'O': [],
        'F': [],
        'H': []
    }
    
    for symbols in symbols_list:
        total = len(symbols)
        counts['total'].append(total)
        counts['C'].append(symbols.count('C'))
        counts['N'].append(symbols.count('N'))
        counts['O'].append(symbols.count('O'))
        counts['F'].append(symbols.count('F'))
        counts['H'].append(symbols.count('H'))
    
    return counts


def compute_empirical_cdf(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute empirical cumulative distribution function.
    
    Args:
        data: 1D numpy array of values
    
    Returns:
        x: Sorted unique values
        cdf: Cumulative probabilities at each x value
    """
    sorted_data = np.sort(data)
    x = np.unique(sorted_data)
    cdf = np.array([np.mean(sorted_data <= val) for val in x])
    return x, cdf


def compute_ks_statistic(real_data: np.ndarray, generated_data: np.ndarray) -> float:
    """
    Compute Kolmogorov-Smirnov-like statistic: 1 - KSD
    
    KSD = max_x |F_Y(x) - F_QM9(x)|
    Returns 1 - KSD, which is 1 when distributions agree perfectly, 0 when they don't overlap.
    
    Args:
        real_data: 1D numpy array of real molecule values
        generated_data: 1D numpy array of generated molecule values
    
    Returns:
        1 - KSD statistic (higher is better, range [0, 1])
        Returns np.nan if either dataset is empty or all NaN
    """
    # Filter out NaN values
    real_valid = ~np.isnan(real_data)
    gen_valid = ~np.isnan(generated_data)
    real_clean = real_data[real_valid]
    gen_clean = generated_data[gen_valid]
    
    if len(real_clean) == 0 or len(gen_clean) == 0:
        return np.nan
    
    # Compute empirical CDFs
    x_real, cdf_real = compute_empirical_cdf(real_clean)
    x_gen, cdf_gen = compute_empirical_cdf(gen_clean)
    
    # Combine all x values and sort
    all_x = np.unique(np.concatenate([x_real, x_gen]))
    all_x = np.sort(all_x)
    
    # Interpolate CDFs at all x values
    cdf_real_interp = np.array([np.mean(real_clean <= x_val) for x_val in all_x])
    cdf_gen_interp = np.array([np.mean(gen_clean <= x_val) for x_val in all_x])
    
    # Compute maximum difference
    ksd = np.max(np.abs(cdf_real_interp - cdf_gen_interp))
    
    return 1.0 - ksd


def plot_distribution_cdfs(real_data_dict: dict, gen_data_dict: dict, 
                           output_dir: str, ks_stats: Optional[dict] = None,
                           plot_type: str = 'atom_counts'):
    """
    Plot marginal distributions with CDFs for atom counts or molecular properties.
    
    Args:
        real_data_dict: Dictionary of real molecule data (atom counts or properties)
        gen_data_dict: Dictionary of generated molecule data
        output_dir: Directory to save plots
        ks_stats: Optional dictionary of KS statistics
        plot_type: 'atom_counts' or 'properties' (affects labels and layout)
    """
    if plot_type == 'atom_counts':
        keys = ['total', 'C', 'N', 'O', 'F', 'H']
        labels = {
            'total': 'Total Atoms',
            'C': 'Carbon',
            'N': 'Nitrogen',
            'O': 'Oxygen',
            'F': 'Fluorine',
            'H': 'Hydrogen'
        }
        xlabel_prefix = 'Number of'
        title_prefix = 'Atom Count'
        output_file = os.path.join(output_dir, 'atom_count_distributions.png')
        figsize = (18, 12)
        nrows, ncols = 2, 3
    else:  # properties
        keys = ['mw', 'logp', 'hbd', 'hba']
        labels = {
            'mw': 'Molecular Weight',
            'logp': 'LogP',
            'hbd': 'H-Bond Donors',
            'hba': 'H-Bond Acceptors'
        }
        xlabel_prefix = ''
        title_prefix = 'Molecular Property'
        output_file = os.path.join(output_dir, 'molecular_property_distributions.png')
        figsize = (16, 10)
        nrows, ncols = 2, 2
    
    # Create figure with subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()
    
    for idx, key in enumerate(keys):
        ax = axes[idx]
        
        real_data = np.array(real_data_dict[key])
        gen_data = np.array(gen_data_dict[key])
        
        # Filter out NaN values
        real_valid = ~np.isnan(real_data)
        gen_valid = ~np.isnan(gen_data)
        real_data_clean = real_data[real_valid]
        gen_data_clean = gen_data[gen_valid]
        
        if len(real_data_clean) == 0 or len(gen_data_clean) == 0:
            ax.text(0.5, 0.5, 'No valid data', transform=ax.transAxes,
                   ha='center', va='center', fontsize=12)
            ax.set_title(f'{labels[key]} Distribution', fontsize=12, fontweight='bold')
            continue
        
        # Compute CDFs
        x_real, cdf_real = compute_empirical_cdf(real_data_clean)
        x_gen, cdf_gen = compute_empirical_cdf(gen_data_clean)
        
        # Plot CDFs
        ax.plot(x_real, cdf_real, label=f'QM9 (n={len(real_data_clean)})', 
               linewidth=2, color='#2E86AB', alpha=0.8)
        ax.plot(x_gen, cdf_gen, label=f'Generated (n={len(gen_data_clean)})', 
               linewidth=2, color='#A23B72', alpha=0.8, linestyle='--')
        
        # Add KS statistic if provided
        if ks_stats and key in ks_stats:
            ks_val = ks_stats[key]
            ax.text(0.05, 0.95, f'1-KSD = {ks_val:.3f}', 
                   transform=ax.transAxes, fontsize=11,
                   verticalalignment='top', bbox=dict(boxstyle='round', 
                   facecolor='wheat', alpha=0.5))
        
        xlabel = f'{xlabel_prefix} {labels[key]}'.strip()
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel('Cumulative Probability', fontsize=11)
        ax.set_title(f'{labels[key]} Distribution', 
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
    
    # Hide unused subplots
    for idx in range(len(keys), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'{title_prefix} Marginal Distributions (CDFs)', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {plot_type} distribution plot to {output_file}")


def plot_atom_count_distributions(real_counts: dict, gen_counts: dict, 
                                  output_dir: str, ks_stats: Optional[dict] = None):
    """
    Plot marginal distributions of atom counts with CDFs.
    
    Args:
        real_counts: Dictionary of atom counts for real molecules
        gen_counts: Dictionary of atom counts for generated molecules
        output_dir: Directory to save plots
        ks_stats: Optional dictionary of KS statistics for each atom type
    """
    plot_distribution_cdfs(real_counts, gen_counts, output_dir, ks_stats, plot_type='atom_counts')


def plot_distribution_comparison(real_embedding: np.ndarray, generated_embedding: np.ndarray,
                                out_file: str, title: str = "Molecule Distribution Comparison",
                                real_num_atoms: Optional[List[int]] = None,
                                gen_num_atoms: Optional[List[int]] = None):
    """
    Plot UMAP embeddings comparing real vs generated molecule distributions.
    
    Args:
        real_embedding: UMAP embedding of real molecules, shape (N, 2)
        generated_embedding: UMAP embedding of generated molecules, shape (M, 2)
        out_file: Output file path for the plot
        title: Plot title
        real_num_atoms: Optional list of number of atoms for real molecules (for coloring)
        gen_num_atoms: Optional list of number of atoms for generated molecules (for coloring)
    """
    # Create side-by-side comparison plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Left: Real (QM9) only
    if real_num_atoms is not None and len(real_num_atoms) == len(real_embedding):
        scatter1 = ax1.scatter(
            real_embedding[:, 0], real_embedding[:, 1], 
            alpha=0.6, s=25, 
            c=real_num_atoms,
            cmap='Blues', edgecolors='none'
        )
        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_label('# Atoms', fontsize=10)
    else:
        ax1.scatter(real_embedding[:, 0], real_embedding[:, 1], 
                   alpha=0.6, s=25, color='#2E86AB', edgecolors='none')
    ax1.set_xlabel('UMAP Dimension 1', fontsize=11)
    ax1.set_ylabel('UMAP Dimension 2', fontsize=11)
    ax1.set_title(f'Real (QM9) (n={len(real_embedding)})', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Middle: Generated only
    if gen_num_atoms is not None and len(gen_num_atoms) == len(generated_embedding):
        scatter2 = ax2.scatter(
            generated_embedding[:, 0], generated_embedding[:, 1], 
            alpha=0.6, s=25,
            c=gen_num_atoms,
            cmap='Reds', edgecolors='none'
        )
        cbar2 = plt.colorbar(scatter2, ax=ax2)
        cbar2.set_label('# Atoms', fontsize=10)
    else:
        ax2.scatter(generated_embedding[:, 0], generated_embedding[:, 1], 
                   alpha=0.6, s=25, color='#A23B72', edgecolors='none')
    ax2.set_xlabel('UMAP Dimension 1', fontsize=11)
    ax2.set_ylabel('UMAP Dimension 2', fontsize=11)
    ax2.set_title(f'Generated (n={len(generated_embedding)})', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Right: Overlay comparison
    ax3.scatter(real_embedding[:, 0], real_embedding[:, 1], 
               alpha=0.4, s=20, label=f'Real (n={len(real_embedding)})', 
               color='#2E86AB', edgecolors='none', marker='o')
    ax3.scatter(generated_embedding[:, 0], generated_embedding[:, 1], 
               alpha=0.5, s=20, label=f'Generated (n={len(generated_embedding)})', 
               color='#A23B72', edgecolors='none', marker='^')
    ax3.set_xlabel('UMAP Dimension 1', fontsize=11)
    ax3.set_ylabel('UMAP Dimension 2', fontsize=11)
    ax3.set_title('Overlay Comparison', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10, loc='best', framealpha=0.9)
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved distribution comparison plot to {out_file}")


def evaluate_molecule_distributions(
    real_symbols: List[List[str]],
    real_positions: List[np.ndarray],
    generated_symbols: List[List[str]],
    generated_positions: List[np.ndarray],
    output_dir: str,
    n_samples: Optional[int] = None,
    n_real_samples: Optional[int] = None,
    n_gen_samples: Optional[int] = None,
    fingerprint_radius: int = 2,
    fingerprint_bits: int = 2048,
    real_smiles: Optional[List[str]] = None,
    generated_smiles: Optional[List[str]] = None,
    fingerprint_type: str = 'rdkit',
    random_seed: Optional[int] = None,
    filter_invalid: bool = True,
    ks_only: bool = False,
    precomputed_real: Optional[dict] = None,
):
    """
    Main evaluation function: Compare real vs generated molecule distributions.
    Based on the paper's evaluation methodology using OpenBabel pipeline.
    
    Args:
        real_symbols: List of lists of atomic symbols for real molecules
        real_positions: List of numpy arrays of atomic positions for real molecules
        generated_symbols: List of lists of atomic symbols for generated molecules
        generated_positions: List of numpy arrays of atomic positions for generated molecules
        output_dir: Directory to save plots
        n_samples: Number of samples to use for both (default None, uses n_real_samples/n_gen_samples)
        n_real_samples: Number of real samples to use (default None = use all available)
        n_gen_samples: Number of generated samples to use (default None = use all available)
        fingerprint_radius: Radius for Morgan fingerprint
        fingerprint_bits: Number of bits in fingerprint
        real_smiles: Optional list of SMILES strings for real molecules (not used, kept for compatibility)
        generated_smiles: Optional list of SMILES strings for generated molecules (not used, kept for compatibility)
        fingerprint_type: Type of fingerprint ('morgan' or 'rdkit')
        random_seed: Random seed for reproducible sampling and UMAP (default 42)
        filter_invalid: If True, filter out invalid and not-fully-connected molecules (default True)
        ks_only: If True, skip UMAP and plots; only compute and save KS statistics (faster)
        precomputed_real: Optional dict with 'properties', 'fps', 'valid', 'atom_counts' to skip
            real-side computation (for batch evaluation)
    
    Returns:
        Dictionary containing evaluation results including fingerprints, embeddings, KS statistics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle backward compatibility: if n_samples is provided, use it for both
    if n_samples is not None:
        if n_real_samples is None:
            n_real_samples = n_samples
        if n_gen_samples is None:
            n_gen_samples = n_samples
    
    # Set random seed for reproducible sampling
    rng = np.random.RandomState(random_seed) if random_seed is not None else np.random
    
    # Sample real molecules if n_real_samples is specified
    if n_real_samples is not None and len(real_symbols) > n_real_samples:
        indices = rng.choice(len(real_symbols), n_real_samples, replace=False)
        indices = np.sort(indices)  # Sort for consistent ordering
        real_symbols = [real_symbols[i] for i in indices]
        real_positions = [real_positions[i] for i in indices]
        if real_smiles:
            real_smiles = [real_smiles[i] for i in indices]
    
    # Sample generated molecules if n_gen_samples is specified
    if n_gen_samples is not None and len(generated_symbols) > n_gen_samples:
        indices = rng.choice(len(generated_symbols), n_gen_samples, replace=False)
        indices = np.sort(indices)  # Sort for consistent ordering
        generated_symbols = [generated_symbols[i] for i in indices]
        generated_positions = [generated_positions[i] for i in indices]
        if generated_smiles:
            generated_smiles = [generated_smiles[i] for i in indices]
    
    # 1. Compute molecular properties (skip if precomputed)
    if precomputed_real is not None:
        real_properties = precomputed_real['properties']
        real_fps = precomputed_real['fps']
        real_valid = precomputed_real['valid']
        real_fp_stats = precomputed_real.get('fp_stats', {})
        print("Using precomputed real molecules data")
    else:
        print("\n" + "="*50)
        print("1. Computing molecular properties...")
        print("="*50)
        print("Computing properties for real molecules...")
        real_properties = compute_molecular_properties(real_symbols, real_positions, filter_invalid=filter_invalid)
    
    # 2. Compute fingerprints (skip real if precomputed)
    if precomputed_real is None:
        print("\n" + "="*50)
        print("2. Computing molecular fingerprints...")
        print("="*50)
        print(f"Using {fingerprint_type} fingerprints...")
        print(f"Computing fingerprints for {len(real_symbols)} real molecules...")
        real_fps, real_valid, real_fp_stats = compute_molecular_fingerprints(
            real_symbols, real_positions,
            radius=fingerprint_radius, n_bits=fingerprint_bits,
            fingerprint_type=fingerprint_type,
            filter_invalid=filter_invalid,
            return_stats=True
        )
        print(f"Valid real molecules: {len(real_fps)}")
    
    print("Computing properties for generated molecules...")
    gen_properties = compute_molecular_properties(generated_symbols, generated_positions, filter_invalid=filter_invalid)
    print(f"Computing fingerprints for {len(generated_symbols)} generated molecules...")
    generated_fps, gen_valid, gen_fp_stats = compute_molecular_fingerprints(
        generated_symbols, generated_positions,
        radius=fingerprint_radius, n_bits=fingerprint_bits,
        fingerprint_type=fingerprint_type,
        filter_invalid=filter_invalid,
        return_stats=True
    )
    print("\nPre-filter stats (all samples):")
    if real_fp_stats:
        print(f"  QM9 total: {real_fp_stats.get('total', '?')} (failed={real_fp_stats.get('failed', 0)})")
    print(f"  Generated total: {gen_fp_stats['total']} (failed={gen_fp_stats['failed']})")
    if filter_invalid and real_fp_stats:
        print("\nFiltering summary (after xyz→sdf→RDKit):")
        print(f"  QM9 valid: {real_fp_stats.get('valid', '?')} / {real_fp_stats.get('total', '?')} "
              f"(invalid={real_fp_stats.get('invalid', 0)}, not_connected={real_fp_stats.get('not_connected', 0)})")
        print(f"  Generated valid: {gen_fp_stats['valid']} / {gen_fp_stats['total']} "
              f"(invalid={gen_fp_stats['invalid']}, not_connected={gen_fp_stats['not_connected']})")
    print(f"Valid generated molecules: {len(generated_fps)}")
    
    # 3. Compute atom counts on filtered molecules only (for consistency)
    print("\n" + "="*50)
    print("3. Computing atom count distributions (on filtered molecules)...")
    print("="*50)
    # Filter symbols to match valid molecules from fingerprint computation
    if filter_invalid:
        real_symbols_filtered = [real_symbols[i] for i in real_valid]
        gen_symbols_filtered = [generated_symbols[i] for i in gen_valid]
    else:
        # If not filtering, use all molecules
        real_symbols_filtered = real_symbols
        gen_symbols_filtered = generated_symbols
    
    real_atom_counts = compute_atom_counts(real_symbols_filtered)
    gen_atom_counts = compute_atom_counts(gen_symbols_filtered)
    
    # 4. Compute Kolmogorov-Smirnov statistics for all metrics
    print("\n" + "="*50)
    print("4. Computing Kolmogorov-Smirnov statistics (1-KSD)...")
    print("="*50)
    ks_stats = {}
    
    # Atom counts
    atom_types = ['total', 'C', 'N', 'O', 'F', 'H']
    print("\nAtom counts:")
    for atom_type in atom_types:
        real_data = np.array(real_atom_counts[atom_type])
        gen_data = np.array(gen_atom_counts[atom_type])
        ks_val = compute_ks_statistic(real_data, gen_data)
        ks_stats[atom_type] = ks_val
        print(f"  {atom_type:>5}: 1-KSD = {ks_val:.4f}")
    
    # Molecular properties
    property_types = ['mw', 'logp', 'hbd', 'hba']
    print("\nMolecular properties:")
    for prop_type in property_types:
        real_data = np.array(real_properties[prop_type])
        gen_data = np.array(gen_properties[prop_type])
        ks_val = compute_ks_statistic(real_data, gen_data)
        ks_stats[prop_type] = ks_val
        if not np.isnan(ks_val):
            print(f"  {prop_type:>5}: 1-KSD = {ks_val:.4f}")
        else:
            print(f"  {prop_type:>5}: 1-KSD = NaN (insufficient valid data)")
    
    # Plot distributions (skipped if ks_only)
    if not ks_only:
        print("\nPlotting atom count marginal distributions...")
        plot_atom_count_distributions(real_atom_counts, gen_atom_counts, output_dir, ks_stats)
        
        print("\nPlotting molecular property distributions...")
        plot_distribution_cdfs(real_properties, gen_properties, output_dir, ks_stats, plot_type='properties')
    
    # 5. Compute UMAP embedding (skipped if ks_only)
    if ks_only:
        real_embedding = np.zeros((len(real_fps), 2))  # Dummy for return dict
        generated_embedding = np.zeros((len(generated_fps), 2))
    else:
        print("\n" + "="*50)
        print("5. Computing UMAP embedding...")
        print("="*50)
        
        # Combine for joint UMAP fitting (better comparison)
        all_fps = np.vstack([real_fps, generated_fps])
        all_embedding = compute_umap_embedding(all_fps, random_state=random_seed)
        
        # Split back
        real_embedding = all_embedding[:len(real_fps)]
        generated_embedding = all_embedding[len(real_fps):]
        
        # Plot UMAP comparison
        output_file = os.path.join(output_dir, 'distribution_comparison_umap.png')
        real_num_atoms = [len(s) for s in real_symbols]
        gen_num_atoms = [len(s) for s in generated_symbols]
        plot_distribution_comparison(real_embedding, generated_embedding, output_file,
                                    real_num_atoms=real_num_atoms, gen_num_atoms=gen_num_atoms)
    
    # Print summary statistics (skip embedding details if ks_only)
    print("\n" + "="*50)
    print("Summary Statistics:")
    print("="*50)
    print(f"Real molecules: {len(real_fps)} valid out of {len(real_symbols)}")
    print(f"Generated molecules: {len(generated_fps)} valid out of {len(generated_symbols)}")
    if not ks_only:
        print(f"\nReal embedding range: X=[{real_embedding[:, 0].min():.2f}, {real_embedding[:, 0].max():.2f}], "
              f"Y=[{real_embedding[:, 1].min():.2f}, {real_embedding[:, 1].max():.2f}]")
        print(f"Generated embedding range: X=[{generated_embedding[:, 0].min():.2f}, {generated_embedding[:, 0].max():.2f}], "
              f"Y=[{generated_embedding[:, 1].min():.2f}, {generated_embedding[:, 1].max():.2f}]")
    
    # Print summary table (matching paper format)
    print("\n" + "="*70)
    print("Distribution Agreement (1-KSD, higher is better)")
    print("="*70)
    print(f"{'Metric':<12} {'1-KSD':>10}")
    print("-" * 70)
    
    # Print in paper order: mw, logp, hbd, hba, C, H, O, N, F, atoms
    paper_order = ['mw', 'logp', 'hbd', 'hba', 'C', 'H', 'O', 'N', 'F', 'total']
    paper_labels = {
        'mw': 'mw', 'logp': 'logp', 'hbd': 'hbd', 'hba': 'hba',
        'C': 'C', 'H': 'H', 'O': 'O', 'N': 'N', 'F': 'F', 'total': 'atoms'
    }
    
    valid_ks_values = []
    for metric in paper_order:
        if metric in ks_stats:
            ks_val = ks_stats[metric]
            label = paper_labels.get(metric, metric)
            if not np.isnan(ks_val):
                print(f"{label:<12} {ks_val:>10.4f}")
                valid_ks_values.append(ks_val)
            else:
                print(f"{label:<12} {'NaN':>10}")
    
    if valid_ks_values:
        avg_ks = np.mean(valid_ks_values)
        print("-" * 70)
        print(f"{'Average':<12} {avg_ks:>10.4f}")
    
    # Save summary table to file
    summary_file = os.path.join(output_dir, 'ks_statistics_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("Distribution Agreement (1-KSD, higher is better)\n")
        f.write("="*70 + "\n")
        f.write(f"{'Metric':<12} {'1-KSD':>10}\n")
        f.write("-" * 70 + "\n")
        for metric in paper_order:
            if metric in ks_stats:
                ks_val = ks_stats[metric]
                label = paper_labels.get(metric, metric)
                if not np.isnan(ks_val):
                    f.write(f"{label:<12} {ks_val:>10.4f}\n")
        if valid_ks_values:
            f.write("-" * 70 + "\n")
            f.write(f"{'Average':<12} {avg_ks:>10.4f}\n")
    print(f"\nSummary saved to {summary_file}")
    
    return {
        'real_fingerprints': real_fps,
        'generated_fingerprints': generated_fps,
        'real_embedding': real_embedding,
        'generated_embedding': generated_embedding,
        'ks_statistics': ks_stats,
        'real_atom_counts': real_atom_counts,
        'generated_atom_counts': gen_atom_counts,
        'real_properties': real_properties,
        'generated_properties': gen_properties
    }


def extract_molecules_from_samples(samples, tokenizer):
    """
    Extract symbols and positions from SamplingResult objects.
    
    Args:
        samples: List of SamplingResult objects from euclidean_sampling, or a single SamplingResult
        tokenizer: VocabTokenizer for decoding atomic symbols
    
    Returns:
        symbols_list: List of lists of atomic symbols
        positions_list: List of numpy arrays of positions
    """
    symbols_list = []
    positions_list = []
    
    # Handle both single sample and list of samples
    if hasattr(samples, 'xt'):  # Single SamplingResult
        samples = [samples]
    
    for sample in samples:
        # Get data
        yt = sample.yt.cpu()  # [seq_len] or [batch, seq_len]
        mask_t = sample.mask_t.cpu()  # [seq_len] or [batch, seq_len]
        xt = sample.xt.cpu()  # [seq_len, 3] or [batch, seq_len, 3]
        
        # Handle batch dimension
        if yt.dim() == 2:
            # Batch dimension present
            batch_size = yt.shape[0]
            for b in range(batch_size):
                yt_b = yt[b]
                mask_t_b = mask_t[b]
                xt_b = xt[b]
                symbols, positions = _extract_single_molecule(yt_b, mask_t_b, xt_b, tokenizer)
                if symbols is not None:
                    symbols_list.append(symbols)
                    positions_list.append(positions)
        else:
            # Single molecule
            symbols, positions = _extract_single_molecule(yt, mask_t, xt, tokenizer)
            if symbols is not None:
                symbols_list.append(symbols)
                positions_list.append(positions)
    
    return symbols_list, positions_list


def save_samples_to_json(samples, tokenizer, json_file: str):
    """
    Extract molecules from SamplingResult objects and save to JSON file.
    
    Convenience function that combines extract_molecules_from_samples and save_molecules_to_json.
    
    Args:
        samples: List of SamplingResult objects from euclidean_sampling, or a single SamplingResult
        tokenizer: VocabTokenizer for decoding atomic symbols
        json_file: Output JSON file path
    """
    symbols_list, positions_list = extract_molecules_from_samples(samples, tokenizer)
    save_molecules_to_json(symbols_list, positions_list, json_file)


def _extract_single_molecule(yt, mask_t, xt, tokenizer):
    """Extract a single molecule from tensors."""
    # Filter out padding
    valid_mask = mask_t.bool()
    valid_yt = yt[valid_mask]
    valid_xt = xt[valid_mask]
    
    # Decode to symbols (skip special tokens)
    symbols = []
    for token_id in valid_yt:
        token_val = token_id.item()
        # Skip special tokens
        if token_val in [tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.mask_token_id]:
            continue
        symbol = tokenizer.idx_to_atom.get(token_val)
        if symbol is not None:
            symbols.append(symbol)
    
    # Get positions (skip BOS position if present)
    positions = valid_xt.numpy()
    
    # Align: if we have BOS at start, skip it
    if len(positions) > len(symbols):
        # Likely BOS/EOS padding, take middle portion
        start_idx = (len(positions) - len(symbols)) // 2
        positions = positions[start_idx:start_idx + len(symbols)]
    elif len(positions) < len(symbols):
        # Pad positions if needed
        padding = np.zeros((len(symbols) - len(positions), 3))
        positions = np.vstack([positions, padding])
    
    if len(symbols) > 0 and len(positions) == len(symbols):
        return symbols, positions
    return None, None


def extract_molecules_from_qm9_dataset(dataset, n_samples: int = 10000, random_seed: int = 42):
    """
    Extract molecules from QM9 dataset.
    
    Args:
        dataset: QM9Dataset instance
        n_samples: Number of samples to extract
        random_seed: Random seed for reproducible sampling (default 42)
    
    Returns:
        symbols_list: List of lists of atomic symbols
        positions_list: List of numpy arrays of positions
        smiles_list: List of SMILES strings (if available)
    """
    symbols_list = []
    positions_list = []
    smiles_list = []
    
    n_samples = min(n_samples, len(dataset))
    # Set random seed for reproducible sampling
    rng = np.random.RandomState(random_seed)
    indices = rng.choice(len(dataset), n_samples, replace=False)
    # Sort indices to ensure consistent ordering
    indices = np.sort(indices)
    
    for idx in indices:
        # Get raw data from underlying dataset to access SMILES
        raw_data = dataset.qm9_dataset[idx]
        smiles = raw_data.get('canonical_smiles') or raw_data.get('smiles')
        
        # Get processed data
        data = dataset[idx]
        mask = data['mask'].numpy()  # Boolean mask indicating valid positions
        
        # Extract symbols using mask
        y_tokens = data['y']  # Tokenized atomic symbols
        symbols = []
        for i, token in enumerate(y_tokens):
            if mask[i]:  # Only process valid (non-padded) positions
                token_id = token.item()
                if token_id not in [dataset.tokenizer.pad_token_id, 
                                   dataset.tokenizer.bos_token_id, 
                                   dataset.tokenizer.eos_token_id,
                                   dataset.tokenizer.mask_token_id]:
                    symbol = dataset.tokenizer.idx_to_atom.get(token_id)
                    if symbol is not None:
                        symbols.append(symbol)
        
        # Extract positions using mask
        positions = data['x'].numpy()  # [max_length, 3]
        valid_positions = positions[mask]  # Filter by mask
        
        # Align positions with symbols (should match since both use same mask)
        if len(symbols) > 0 and len(valid_positions) > 0:
            # Take the first len(symbols) positions
            if len(valid_positions) >= len(symbols):
                positions_final = valid_positions[:len(symbols)]
            else:
                # Pad if needed (shouldn't happen, but safety check)
                padding = np.zeros((len(symbols) - len(valid_positions), 3))
                positions_final = np.vstack([valid_positions, padding])
            
            if len(symbols) == len(positions_final):
                symbols_list.append(symbols)
                positions_list.append(positions_final)
                smiles_list.append(smiles)
    
    return symbols_list, positions_list, smiles_list


def load_molecules_from_json(json_file: str):
    """
    Load molecules from a JSON file.
    
    Expected JSON format:
    {
        "molecules": [
            {
                "symbols": ["C", "H", "H", "H", "H"],
                "positions": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], ...]
            },
            ...
        ]
    }
    
    Args:
        json_file: Path to JSON file containing molecules
    
    Returns:
        symbols_list: List of lists of atomic symbols
        positions_list: List of numpy arrays of positions
    """
    import json
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    if 'molecules' not in data:
        raise ValueError(f"JSON file must contain a 'molecules' key. Got keys: {list(data.keys())}")
    
    symbols_list = []
    positions_list = []
    
    for i, mol in enumerate(data['molecules']):
        if 'symbols' not in mol or 'positions' not in mol:
            print(f"Warning: Skipping molecule {i} - missing 'symbols' or 'positions'")
            continue
        
        symbols = mol['symbols']
        positions = np.array(mol['positions'])
        
        # Validate shapes
        if len(symbols) != len(positions):
            print(f"Warning: Skipping molecule {i} - symbols length ({len(symbols)}) != positions length ({len(positions)})")
            continue
        
        if positions.shape[1] != 3:
            print(f"Warning: Skipping molecule {i} - positions must be (N, 3), got {positions.shape}")
            continue
        
        symbols_list.append(symbols)
        positions_list.append(positions)
    
    print(f"Loaded {len(symbols_list)} molecules from {json_file}")
    return symbols_list, positions_list


def save_molecules_to_json(symbols_list: List[List[str]], positions_list: List[np.ndarray], 
                           json_file: str):
    """
    Save molecules to a JSON file.
    
    Args:
        symbols_list: List of lists of atomic symbols
        positions_list: List of numpy arrays of positions
        json_file: Output JSON file path
    """
    import json
    
    molecules = []
    for symbols, positions in zip(symbols_list, positions_list):
        molecules.append({
            'symbols': symbols,
            'positions': positions.tolist()  # Convert numpy array to list
        })
    
    data = {'molecules': molecules}
    
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved {len(molecules)} molecules to {json_file}")


# Example usage:
"""
Example usage of the evaluation script:

```python
from evaluate_distribution import (
    evaluate_molecule_distributions, 
    extract_molecules_from_samples, 
    extract_molecules_from_qm9_dataset
)
from custom_datasets.qm9 import QM9Dataset
from utils.tokenizer import VocabTokenizer
from multimodal_interpolant import MultimodalInterpolant
import torch

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = VocabTokenizer(vocab={'H', 'C', 'N', 'O', 'F'})
dataset = QM9Dataset(tokenizer)

# Load your trained model and interpolant
# model = load_your_model(...)
# interpolant = MultimodalInterpolant(...)

# Generate 10k samples (batch in smaller chunks)
# all_samples = []
# for _ in range(50):  # 50 batches of 200 = 10k samples
#     samples = interpolant.euclidean_sampling(model, steps=50, batch_size=200, 
#                                             max_length=dataset.max_length, device=device)
#     all_samples.extend(samples)
# gen_symbols, gen_positions = extract_molecules_from_samples(all_samples, tokenizer)

# Extract 10k real samples
# real_symbols, real_positions = extract_molecules_from_qm9_dataset(dataset, n_samples=10000)

# Evaluate and plot
# results = evaluate_molecule_distributions(
#     real_symbols, real_positions,
#     gen_symbols, gen_positions,
#     output_dir='evaluation_results/',
#     n_samples=10000
# )
```
"""
