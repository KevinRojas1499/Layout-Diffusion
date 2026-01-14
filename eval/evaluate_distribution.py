"""
Evaluation script for comparing generated vs real molecule distributions using UMAP.
Based on the paper's evaluation methodology using molecular fingerprints.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import os

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    import warnings
    # Suppress deprecation warnings for GetMorganFingerprintAsBitVect
    warnings.filterwarnings("ignore", message=".*GetMorganFingerprintAsBitVect.*", category=DeprecationWarning)
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
                                   radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """
    Compute Morgan fingerprints for molecules.
    
    Args:
        symbols_list: List of lists, where each inner list contains atomic symbols
        positions_list: List of numpy arrays, where each array is (N, 3) atomic positions
        smiles_list: Optional list of SMILES strings (preferred over position-based conversion)
        radius: Radius for Morgan fingerprint (default 2)
        n_bits: Number of bits in fingerprint (default 2048)
    
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
            
            # Compute Morgan fingerprint
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
    Uses distance-based bond detection.
    
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
        
        # Try to add bonds based on distance
        # Use a more sophisticated approach: find bonds greedily
        num_atoms = len(symbols)
        bonded_pairs = []
        
        # Get all potential bonds sorted by distance
        potential_bonds = []
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                dist = np.linalg.norm(positions[i] - positions[j])
                if _is_bonded(symbols[i], symbols[j], dist):
                    potential_bonds.append((i, j, dist))
        
        # Sort by distance (shorter bonds first)
        potential_bonds.sort(key=lambda x: x[2])
        
        # Add bonds greedily, avoiding cycles (simple check)
        added_bonds = set()
        atom_degrees = {i: 0 for i in range(num_atoms)}
        
        for i, j, dist in potential_bonds:
            # Check if adding this bond would exceed reasonable valency
            max_valency = _get_max_valency(symbols[i], symbols[j])
            if atom_degrees[i] < max_valency and atom_degrees[j] < max_valency:
                # Simple cycle check: don't add if both atoms already have bonds
                # (this is a heuristic, not perfect)
                if atom_degrees[i] == 0 or atom_degrees[j] == 0 or len(added_bonds) < num_atoms - 1:
                    try:
                        mol.AddBond(i, j, Chem.BondType.SINGLE)
                        added_bonds.add((i, j))
                        atom_degrees[i] += 1
                        atom_degrees[j] += 1
                    except:
                        pass
        
        # If no bonds were added, try a simpler approach: just add all potential bonds
        if len(added_bonds) == 0:
            for i, j, dist in potential_bonds[:num_atoms * 2]:  # Limit to avoid too many bonds
                try:
                    mol.AddBond(i, j, Chem.BondType.SINGLE)
                except:
                    pass
        
        # Try to sanitize molecule
        try:
            Chem.SanitizeMol(mol)
            return mol.GetMol()
        except:
            # If sanitization fails, try to return unsanitized molecule
            # Some fingerprint functions might still work
            try:
                return mol.GetMol()
            except:
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
    
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        metric='jaccard'  # Jaccard distance is good for binary fingerprints
    )
    
    embedding = reducer.fit_transform(fingerprints)
    return embedding


def plot_distribution_comparison(real_embedding: np.ndarray, generated_embedding: np.ndarray,
                                out_file: str, title: str = "Molecule Distribution Comparison"):
    """
    Plot UMAP embeddings comparing real vs generated molecule distributions.
    
    Args:
        real_embedding: UMAP embedding of real molecules, shape (N, 2)
        generated_embedding: UMAP embedding of generated molecules, shape (M, 2)
        out_file: Output file path for the plot
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot real molecules
    ax.scatter(real_embedding[:, 0], real_embedding[:, 1], 
              alpha=0.5, s=20, label='Real (QM9)', color='blue', edgecolors='none')
    
    # Plot generated molecules
    ax.scatter(generated_embedding[:, 0], generated_embedding[:, 1], 
              alpha=0.5, s=20, label='Generated', color='red', edgecolors='none')
    
    ax.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax.set_ylabel('UMAP Dimension 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
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
    generated_smiles: Optional[List[str]] = None
):
    """
    Main evaluation function: Compare real vs generated molecule distributions.
    
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
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for evaluation")
    if not UMAP_AVAILABLE:
        raise ImportError("UMAP is required for evaluation")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample if we have more than n_samples
    if len(real_symbols) > n_samples:
        indices = np.random.choice(len(real_symbols), n_samples, replace=False)
        real_symbols = [real_symbols[i] for i in indices]
        real_positions = [real_positions[i] for i in indices]
        if real_smiles:
            real_smiles = [real_smiles[i] for i in indices]
    
    if len(generated_symbols) > n_samples:
        indices = np.random.choice(len(generated_symbols), n_samples, replace=False)
        generated_symbols = [generated_symbols[i] for i in indices]
        generated_positions = [generated_positions[i] for i in indices]
        if generated_smiles:
            generated_smiles = [generated_smiles[i] for i in indices]
    
    print(f"Computing fingerprints for {len(real_symbols)} real molecules...")
    real_fps, real_valid = compute_molecular_fingerprints(
        real_symbols, real_positions, smiles_list=real_smiles, 
        radius=fingerprint_radius, n_bits=fingerprint_bits
    )
    print(f"Valid real molecules: {len(real_fps)}")
    
    print(f"Computing fingerprints for {len(generated_symbols)} generated molecules...")
    generated_fps, gen_valid = compute_molecular_fingerprints(
        generated_symbols, generated_positions, smiles_list=generated_smiles,
        radius=fingerprint_radius, n_bits=fingerprint_bits
    )
    print(f"Valid generated molecules: {len(generated_fps)}")
    
    # Combine for joint UMAP fitting (better comparison)
    print("Computing UMAP embedding...")
    all_fps = np.vstack([real_fps, generated_fps])
    all_embedding = compute_umap_embedding(all_fps)
    
    # Split back
    real_embedding = all_embedding[:len(real_fps)]
    generated_embedding = all_embedding[len(real_fps):]
    
    # Plot comparison
    output_file = os.path.join(output_dir, 'distribution_comparison_umap.png')
    plot_distribution_comparison(real_embedding, generated_embedding, output_file)
    
    # Compute some statistics
    print("\n" + "="*50)
    print("Distribution Statistics:")
    print("="*50)
    print(f"Real molecules: {len(real_fps)} valid out of {len(real_symbols)}")
    print(f"Generated molecules: {len(generated_fps)} valid out of {len(generated_symbols)}")
    print(f"Real embedding range: X=[{real_embedding[:, 0].min():.2f}, {real_embedding[:, 0].max():.2f}], "
          f"Y=[{real_embedding[:, 1].min():.2f}, {real_embedding[:, 1].max():.2f}]")
    print(f"Generated embedding range: X=[{generated_embedding[:, 0].min():.2f}, {generated_embedding[:, 0].max():.2f}], "
          f"Y=[{generated_embedding[:, 1].min():.2f}, {generated_embedding[:, 1].max():.2f}]")
    
    return {
        'real_fingerprints': real_fps,
        'generated_fingerprints': generated_fps,
        'real_embedding': real_embedding,
        'generated_embedding': generated_embedding
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


def extract_molecules_from_qm9_dataset(dataset, n_samples: int = 10000):
    """
    Extract molecules from QM9 dataset.
    
    Args:
        dataset: QM9Dataset instance
        n_samples: Number of samples to extract
    
    Returns:
        symbols_list: List of lists of atomic symbols
        positions_list: List of numpy arrays of positions
        smiles_list: List of SMILES strings (if available)
    """
    symbols_list = []
    positions_list = []
    smiles_list = []
    
    n_samples = min(n_samples, len(dataset))
    indices = np.random.choice(len(dataset), n_samples, replace=False)
    
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
