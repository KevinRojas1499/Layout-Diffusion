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

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import RDLogger
    import warnings
    # Suppress deprecation warnings for GetMorganFingerprintAsBitVect
    warnings.filterwarnings("ignore", message=".*GetMorganFingerprintAsBitVect.*", category=DeprecationWarning)
    # Suppress RDKit error/warning messages (they're too verbose for invalid molecules)
    RDLogger.DisableLog('rdApp.*')
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available. Install with: conda install -c conda-forge rdkit")

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: UMAP not available. Install with: pip install umap-learn")


def compute_molecular_fingerprints(symbols_list: List[List[str]], positions_list: List[np.ndarray], 
                                   smiles_list: Optional[List[str]] = None,
                                   radius: int = 2, n_bits: int = 2048,
                                   fingerprint_type: str = 'morgan') -> np.ndarray:
    """
    Compute molecular fingerprints for molecules.
    
    Args:
        symbols_list: List of lists, where each inner list contains atomic symbols
        positions_list: List of numpy arrays, where each array is (N, 3) atomic positions
        smiles_list: Optional list of SMILES strings (preferred over position-based conversion)
        radius: Radius for Morgan fingerprint (default 2)
        n_bits: Number of bits in fingerprint (default 2048)
        fingerprint_type: Type of fingerprint ('morgan' or 'rdkit')
    
    Returns:
        numpy array of shape (num_molecules, n_bits) with binary fingerprints
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for fingerprint computation")
    
    fingerprints = []
    valid_indices = []
    failed_count = 0
    
    for idx, (symbols, positions) in enumerate(zip(symbols_list, positions_list)):
        try:
            # Prefer SMILES if available
            smiles = smiles_list[idx] if smiles_list and idx < len(smiles_list) else None
            mol = None
            
            if smiles:
                # Try to create molecule from SMILES
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    # Add 3D coordinates if available
                    try:
                        mol = Chem.AddHs(mol)  # Add hydrogens
                        AllChem.EmbedMolecule(mol, randomSeed=42)
                        AllChem.MMFFOptimizeMolecule(mol)
                    except:
                        pass  # Continue without 3D coordinates
            
            # Fallback to position-based creation if SMILES failed
            if mol is None:
                mol = _positions_to_molecule(symbols, positions)
            
            if mol is None:
                failed_count += 1
                continue
            
            # Compute fingerprint based on type
            if fingerprint_type.lower() == 'rdkit':
                # RDKit fingerprint (standard RDKit topological fingerprint)
                # This is the standard RDKit fingerprint used in the paper
                fp = Chem.RDKFingerprint(mol, maxPath=7, fpSize=n_bits)
            else:
                # Morgan fingerprint (default)
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
            
            fingerprints.append(np.array(fp))
            valid_indices.append(idx)
        except Exception as e:
            failed_count += 1
            if idx < 5:  # Only print first few errors
                print(f"Warning: Failed to compute fingerprint for molecule {idx}: {e}")
            continue
    
    if len(fingerprints) == 0:
        raise ValueError(f"No valid fingerprints computed (failed for all {len(symbols_list)} molecules)")
    
    if failed_count > 0:
        print(f"Successfully computed {len(fingerprints)} fingerprints ({failed_count} failed)")
    
    return np.array(fingerprints), valid_indices


def _positions_to_molecule(symbols: List[str], positions: np.ndarray):
    """
    Convert atomic symbols and positions to RDKit molecule object.
    Uses distance-based bond detection with strict valency checking.
    
    Args:
        symbols: List of atomic symbols
        positions: numpy array of shape (N, 3) with atomic positions
    
    Returns:
        RDKit molecule object or None if conversion fails
    """
    if not RDKIT_AVAILABLE:
        return None
    
    if len(symbols) == 0 or len(positions) == 0:
        return None
    
    try:
        # Create molecule from symbols and positions
        mol = Chem.RWMol()
        
        # Add atoms
        for symbol in symbols:
            try:
                atom = Chem.Atom(symbol)
                mol.AddAtom(atom)
            except:
                return None  # Invalid atom symbol
        
        if mol.GetNumAtoms() == 0:
            return None
        
        # Set 3D coordinates
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i, pos in enumerate(positions):
            if i < mol.GetNumAtoms():
                conf.SetAtomPosition(i, tuple(pos))
        mol.AddConformer(conf)
        
        # Get maximum valencies for each atom
        valencies = {
            'H': 1, 'C': 4, 'N': 3, 'O': 2, 'F': 1,
            'S': 2, 'Cl': 1, 'P': 3, 'Br': 1
        }
        max_valencies = [valencies.get(sym, 4) for sym in symbols]
        
        # Get all potential bonds sorted by distance
        potential_bonds = []
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                dist = np.linalg.norm(positions[i] - positions[j])
                if _is_bonded(symbols[i], symbols[j], dist):
                    potential_bonds.append((i, j, dist))
        
        # Sort by distance (shorter bonds first)
        potential_bonds.sort(key=lambda x: x[2])
        
        # Add bonds greedily with strict valency checking
        added_bonds = set()
        atom_degrees = [0] * len(symbols)
        
        for i, j, dist in potential_bonds:
            # Strict check: both atoms must have available valency
            if atom_degrees[i] < max_valencies[i] and atom_degrees[j] < max_valencies[j]:
                # Check if adding this bond would create a reasonable structure
                # (avoid creating too many bonds to the same atom)
                    try:
                        mol.AddBond(i, j, Chem.BondType.SINGLE)
                        added_bonds.add((i, j))
                        atom_degrees[i] += 1
                        atom_degrees[j] += 1
                    except:
                    # Bond addition failed, skip it
                        pass
        
        # If no bonds were added, the molecule is likely invalid
        if len(added_bonds) == 0:
            return None
        
        # Try to sanitize molecule - this will fail if valencies are invalid
        try:
            Chem.SanitizeMol(mol)
            return mol.GetMol()
        except:
            # Sanitization failed - molecule has invalid structure
                return None
            
    except Exception as e:
        return None


def _get_max_valency(atom1: str, atom2: str) -> int:
    """Get maximum valency for bond detection."""
    # Common valencies
    valencies = {
        'H': 1, 'C': 4, 'N': 3, 'O': 2, 'F': 1,
        'S': 2, 'Cl': 1, 'P': 3, 'Br': 1
    }
    return max(valencies.get(atom1, 4), valencies.get(atom2, 4))


def _is_bonded(atom1: str, atom2: str, distance: float) -> bool:
    """Check if two atoms are likely bonded based on distance."""
    # Covalent radii (Angstroms)
    radii = {
        'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57,
        'S': 1.05, 'Cl': 1.02, 'P': 1.07, 'Br': 1.20
    }
    
    r1 = radii.get(atom1, 0.7)
    r2 = radii.get(atom2, 0.7)
    threshold = (r1 + r2) * 1.3  # 30% tolerance
    
    return distance < threshold


def compute_umap_embedding(fingerprints: np.ndarray, n_components: int = 2, 
                          n_neighbors: int = 15, min_dist: float = 0.1, 
                          random_state: int = 42) -> np.ndarray:
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
    if not UMAP_AVAILABLE:
        raise ImportError("UMAP is required for embedding computation")
    
    # Set numpy random seed for additional reproducibility
    # (UMAP uses numpy random internally)
    np.random.seed(random_state)
    
    # Suppress UMAP warnings about Jaccard metric and n_jobs
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*gradient function is not yet implemented.*')
        warnings.filterwarnings('ignore', message='.*n_jobs value.*overridden.*')
        
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
            metric='jaccard',  # Jaccard distance is good for binary fingerprints
            verbose=False,  # Suppress UMAP output for cleaner logs
            n_jobs=1  # Explicitly set to avoid warning when random_state is set
        )
        
        embedding = reducer.fit_transform(fingerprints)
    
    return embedding


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
    """
    # Compute empirical CDFs
    x_real, cdf_real = compute_empirical_cdf(real_data)
    x_gen, cdf_gen = compute_empirical_cdf(generated_data)
    
    # Combine all x values and sort
    all_x = np.unique(np.concatenate([x_real, x_gen]))
    all_x = np.sort(all_x)
    
    # Interpolate CDFs at all x values
    cdf_real_interp = np.array([np.mean(real_data <= x_val) for x_val in all_x])
    cdf_gen_interp = np.array([np.mean(generated_data <= x_val) for x_val in all_x])
    
    # Compute maximum difference
    ksd = np.max(np.abs(cdf_real_interp - cdf_gen_interp))
    
    return 1.0 - ksd


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
    atom_types = ['total', 'C', 'N', 'O', 'F', 'H']
    atom_labels = {
        'total': 'Total Atoms',
        'C': 'Carbon',
        'N': 'Nitrogen',
        'O': 'Oxygen',
        'F': 'Fluorine',
        'H': 'Hydrogen'
    }
    
    # Create figure with subplots: 2 rows, 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, atom_type in enumerate(atom_types):
        ax = axes[idx]
        
        real_data = np.array(real_counts[atom_type])
        gen_data = np.array(gen_counts[atom_type])
        
        # Compute CDFs
        x_real, cdf_real = compute_empirical_cdf(real_data)
        x_gen, cdf_gen = compute_empirical_cdf(gen_data)
        
        # Plot CDFs
        ax.plot(x_real, cdf_real, label=f'QM9 (n={len(real_data)})', 
               linewidth=2, color='#2E86AB', alpha=0.8)
        ax.plot(x_gen, cdf_gen, label=f'Generated (n={len(gen_data)})', 
               linewidth=2, color='#A23B72', alpha=0.8, linestyle='--')
        
        # Add KS statistic if provided
        if ks_stats and atom_type in ks_stats:
            ks_val = ks_stats[atom_type]
            ax.text(0.05, 0.95, f'1-KSD = {ks_val:.3f}', 
                   transform=ax.transAxes, fontsize=11,
                   verticalalignment='top', bbox=dict(boxstyle='round', 
                   facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel(f'Number of {atom_labels[atom_type]}', fontsize=11)
        ax.set_ylabel('Cumulative Probability', fontsize=11)
        ax.set_title(f'{atom_labels[atom_type]} Count Distribution', 
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
    
    plt.suptitle('Atom Count Marginal Distributions (CDFs)', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'atom_count_distributions.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved atom count distribution plot to {output_file}")


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
    n_samples: int = 10000,
    fingerprint_radius: int = 2,
    fingerprint_bits: int = 2048,
    real_smiles: Optional[List[str]] = None,
    generated_smiles: Optional[List[str]] = None,
    fingerprint_type: str = 'morgan',
    random_seed: int = 42
):
    """
    Main evaluation function: Compare real vs generated molecule distributions.
    Based on the paper's evaluation methodology.
    
    Args:
        real_symbols: List of lists of atomic symbols for real molecules
        real_positions: List of numpy arrays of atomic positions for real molecules
        generated_symbols: List of lists of atomic symbols for generated molecules
        generated_positions: List of numpy arrays of atomic positions for generated molecules
        output_dir: Directory to save plots
        n_samples: Number of samples to use (default 10000)
        fingerprint_radius: Radius for Morgan fingerprint
        fingerprint_bits: Number of bits in fingerprint
        real_smiles: Optional list of SMILES strings for real molecules
        generated_smiles: Optional list of SMILES strings for generated molecules
        fingerprint_type: Type of fingerprint ('morgan' or 'rdkit')
        random_seed: Random seed for reproducible sampling and UMAP (default 42)
    
    Returns:
        Dictionary containing evaluation results including fingerprints, embeddings, KS statistics
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for evaluation")
    if not UMAP_AVAILABLE:
        raise ImportError("UMAP is required for evaluation")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seed for reproducible sampling
    rng = np.random.RandomState(random_seed)
    
    # Sample if we have more than n_samples
    if len(real_symbols) > n_samples:
        indices = rng.choice(len(real_symbols), n_samples, replace=False)
        indices = np.sort(indices)  # Sort for consistent ordering
        real_symbols = [real_symbols[i] for i in indices]
        real_positions = [real_positions[i] for i in indices]
        if real_smiles:
            real_smiles = [real_smiles[i] for i in indices]
    
    if len(generated_symbols) > n_samples:
        indices = rng.choice(len(generated_symbols), n_samples, replace=False)
        indices = np.sort(indices)  # Sort for consistent ordering
        generated_symbols = [generated_symbols[i] for i in indices]
        generated_positions = [generated_positions[i] for i in indices]
        if generated_smiles:
            generated_smiles = [generated_smiles[i] for i in indices]
    
    # 1. Compute atom count distributions
    print("\n" + "="*50)
    print("1. Computing atom count distributions...")
    print("="*50)
    real_atom_counts = compute_atom_counts(real_symbols)
    gen_atom_counts = compute_atom_counts(generated_symbols)
    
    # Compute Kolmogorov-Smirnov statistics for each atom type
    print("\nComputing Kolmogorov-Smirnov statistics (1-KSD)...")
    ks_stats = {}
    atom_types = ['total', 'C', 'N', 'O', 'F', 'H']
    for atom_type in atom_types:
        real_data = np.array(real_atom_counts[atom_type])
        gen_data = np.array(gen_atom_counts[atom_type])
        ks_val = compute_ks_statistic(real_data, gen_data)
        ks_stats[atom_type] = ks_val
        print(f"  {atom_type:>5}: 1-KSD = {ks_val:.4f}")
    
    # Plot atom count distributions
    print("\nPlotting atom count marginal distributions...")
    plot_atom_count_distributions(real_atom_counts, gen_atom_counts, output_dir, ks_stats)
    
    # 2. Compute fingerprints and UMAP
    print("\n" + "="*50)
    print("2. Computing molecular fingerprints and UMAP embedding...")
    print("="*50)
    print(f"Using {fingerprint_type} fingerprints...")
    print(f"Computing fingerprints for {len(real_symbols)} real molecules...")
    real_fps, real_valid = compute_molecular_fingerprints(
        real_symbols, real_positions, smiles_list=real_smiles, 
        radius=fingerprint_radius, n_bits=fingerprint_bits,
        fingerprint_type=fingerprint_type
    )
    print(f"Valid real molecules: {len(real_fps)}")
    
    print(f"Computing fingerprints for {len(generated_symbols)} generated molecules...")
    generated_fps, gen_valid = compute_molecular_fingerprints(
        generated_symbols, generated_positions, smiles_list=generated_smiles,
        radius=fingerprint_radius, n_bits=fingerprint_bits,
        fingerprint_type=fingerprint_type
    )
    print(f"Valid generated molecules: {len(generated_fps)}")
    
    # Combine for joint UMAP fitting (better comparison)
    print("Computing UMAP embedding...")
    all_fps = np.vstack([real_fps, generated_fps])
    # Use the same random seed for UMAP to ensure reproducibility
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
    
    # Print summary statistics
    print("\n" + "="*50)
    print("Summary Statistics:")
    print("="*50)
    print(f"Real molecules: {len(real_fps)} valid out of {len(real_symbols)}")
    print(f"Generated molecules: {len(generated_fps)} valid out of {len(generated_symbols)}")
    print(f"\nReal embedding range: X=[{real_embedding[:, 0].min():.2f}, {real_embedding[:, 0].max():.2f}], "
          f"Y=[{real_embedding[:, 1].min():.2f}, {real_embedding[:, 1].max():.2f}]")
    print(f"Generated embedding range: X=[{generated_embedding[:, 0].min():.2f}, {generated_embedding[:, 0].max():.2f}], "
          f"Y=[{generated_embedding[:, 1].min():.2f}, {generated_embedding[:, 1].max():.2f}]")
    
    print("\n" + "="*50)
    print("Kolmogorov-Smirnov Statistics (1-KSD):")
    print("="*50)
    print("Higher values indicate better distribution match (range: 0-1)")
    for atom_type in atom_types:
        print(f"  {atom_type:>5}: {ks_stats[atom_type]:.4f}")
    avg_ks = np.mean(list(ks_stats.values()))
    print(f"\n  Average: {avg_ks:.4f}")
    
    return {
        'real_fingerprints': real_fps,
        'generated_fingerprints': generated_fps,
        'real_embedding': real_embedding,
        'generated_embedding': generated_embedding,
        'ks_statistics': ks_stats,
        'real_atom_counts': real_atom_counts,
        'generated_atom_counts': gen_atom_counts
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
