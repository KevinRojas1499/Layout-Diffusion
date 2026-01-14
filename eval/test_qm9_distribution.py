"""
Test script to plot QM9Dataset distribution using UMAP.
This verifies that the evaluation functions work correctly on real data.
"""

import numpy as np
import matplotlib.pyplot as plt
from evaluate_distribution import (
    extract_molecules_from_qm9_dataset,
    compute_molecular_fingerprints,
    compute_umap_embedding,
    plot_distribution_comparison
)
from custom_datasets.qm9 import QM9Dataset
from utils.tokenizer import VocabTokenizer

def plot_qm9_distribution(n_samples: int = 10000, output_file: str = 'qm9_distribution_test.png'):
    """
    Plot QM9 dataset distribution using UMAP.
    
    Args:
        n_samples: Number of samples to use (default 5000 for quick test)
        output_file: Output file path for the plot
    """
    print("="*60)
    print("Testing QM9 Dataset Distribution Visualization")
    print("="*60)
    
    # Setup tokenizer and dataset
    print(f"\n1. Loading QM9Dataset...")
    tokenizer = VocabTokenizer(vocab={'H', 'C', 'N', 'O', 'F'})
    dataset = QM9Dataset(tokenizer, max_length=30, canonical_order=True)
    print(f"   Dataset size: {len(dataset)}")
    
    # Extract molecules
    print(f"\n2. Extracting {n_samples} molecules from dataset...")
    symbols_list, positions_list, smiles_list = extract_molecules_from_qm9_dataset(dataset, n_samples=n_samples)
    print(f"   Successfully extracted {len(symbols_list)} valid molecules")
    print(f"   SMILES available for {sum(1 for s in smiles_list if s is not None)} molecules")
    
    if len(symbols_list) == 0:
        print("ERROR: No valid molecules extracted!")
        return
    
    # Show some statistics
    print(f"\n3. Molecule statistics:")
    num_atoms = [len(s) for s in symbols_list]
    print(f"   Average number of atoms: {np.mean(num_atoms):.2f}")
    print(f"   Min atoms: {min(num_atoms)}, Max atoms: {max(num_atoms)}")
    
    atom_counts = {}
    for symbols in symbols_list:
        for symbol in symbols:
            atom_counts[symbol] = atom_counts.get(symbol, 0) + 1
    print(f"   Atom distribution: {atom_counts}")
    
    # Compute fingerprints
    print(f"\n4. Computing molecular fingerprints...")
    try:
        fingerprints, valid_indices = compute_molecular_fingerprints(
            symbols_list, positions_list, smiles_list=smiles_list, radius=2, n_bits=2048
        )
        print(f"   Computed {len(fingerprints)} valid fingerprints")
        print(f"   Fingerprint shape: {fingerprints.shape}")
        print(f"   Fingerprint sparsity: {(fingerprints == 0).sum() / fingerprints.size * 100:.2f}%")
    except Exception as e:
        print(f"   ERROR computing fingerprints: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Compute UMAP embedding
    print(f"\n5. Computing UMAP embedding...")
    try:
        embedding = compute_umap_embedding(
            fingerprints, 
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            random_state=42
        )
        print(f"   Embedding shape: {embedding.shape}")
        print(f"   Embedding range: X=[{embedding[:, 0].min():.2f}, {embedding[:, 0].max():.2f}], "
              f"Y=[{embedding[:, 1].min():.2f}, {embedding[:, 1].max():.2f}]")
    except Exception as e:
        print(f"   ERROR computing UMAP: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Plot distribution
    print(f"\n6. Plotting distribution...")
    try:
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create a scatter plot
        scatter = ax.scatter(
            embedding[:, 0], 
            embedding[:, 1], 
            alpha=0.6, 
            s=20, 
            c=num_atoms[:len(embedding)],  # Color by number of atoms
            cmap='viridis',
            edgecolors='none'
        )
        
        ax.set_xlabel('UMAP Dimension 1', fontsize=12)
        ax.set_ylabel('UMAP Dimension 2', fontsize=12)
        ax.set_title(f'QM9 Dataset Distribution (n={len(embedding)} molecules)', 
                    fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Number of Atoms', fontsize=11)
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   Saved plot to {output_file}")
    except Exception as e:
        print(f"   ERROR plotting: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*60)
    print("SUCCESS: QM9 distribution visualization completed!")
    print("="*60)
    print(f"\nOutput saved to: {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test QM9 dataset distribution visualization')
    parser.add_argument('--n_samples', type=int, default=5000,
                       help='Number of samples to use (default: 5000)')
    parser.add_argument('--output', type=str, default='qm9_distribution_test.png',
                       help='Output file path (default: qm9_distribution_test.png)')
    
    args = parser.parse_args()
    
    plot_qm9_distribution(n_samples=args.n_samples, output_file=args.output)
