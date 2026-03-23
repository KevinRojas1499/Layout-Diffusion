"""
Test script to plot QM9Dataset distribution using UMAP.
Can also compare generated molecules vs QM9 dataset.
"""

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from eval.evaluate_distribution import (
    extract_molecules_from_qm9_dataset,
    compute_molecular_fingerprints,
    compute_umap_embedding,
    plot_distribution_comparison,
    load_molecules_from_json,
    evaluate_molecule_distributions
)
from custom_datasets.qm9 import QM9Dataset
from utils.tokenizer import VocabTokenizer

def plot_qm9_distribution(n_samples: int = 10000, output_file: str = 'qm9_distribution_test.png',
                          generated_file: str = None, comparison_output: str = None,
                          n_real_samples: Optional[int] = None, n_gen_samples: Optional[int] = None,
                          use_all_samples: bool = False, ks_only: bool = False):
    """
    Plot QM9 dataset distribution using UMAP.
    
    Args:
        n_samples: Number of samples to use for both (default 10000, used if n_real_samples/n_gen_samples not set)
        output_file: Output file path for the plot
        generated_file: Path to JSON file with generated molecules (optional)
        comparison_output: Directory to save full comparison results (optional)
        n_real_samples: Number of QM9 samples to use (default None = use n_samples, or all if n_samples not set)
        n_gen_samples: Number of generated samples to use (default None = use all available)
        use_all_samples: If True, use all generated samples (overrides n_gen_samples, but not n_real_samples or n_samples)
    """
    print("="*60)
    if generated_file:
        print("Comparing Generated vs QM9 Dataset Distributions")
    else:
        print("Testing QM9 Dataset Distribution Visualization")
        print("="*60)
    
    # Setup tokenizer and dataset
    print(f"\n1. Loading QM9Dataset...")
    tokenizer = VocabTokenizer(vocab={'H', 'C', 'N', 'O', 'F'})
    dataset = QM9Dataset(tokenizer, max_length=30)
    print(f"   Dataset size: {len(dataset)}")
    
    # Determine how many QM9 samples to use
    qm9_n_samples = n_real_samples if n_real_samples is not None else n_samples
    
    # Extract QM9 molecules
    print(f"\n2. Extracting {qm9_n_samples} molecules from QM9 dataset...")
    # Use fixed random seed for reproducibility
    real_symbols, real_positions, real_smiles = extract_molecules_from_qm9_dataset(
        dataset, n_samples=qm9_n_samples, random_seed=42
    )
    print(f"   Successfully extracted {len(real_symbols)} valid molecules")
    print(f"   SMILES available for {sum(1 for s in real_smiles if s is not None)} molecules")
    
    if len(real_symbols) == 0:
        print("ERROR: No valid molecules extracted from QM9!")
        return
    
    # Load generated molecules if provided
    gen_symbols = None
    gen_positions = None
    if generated_file:
        print(f"\n3. Loading generated molecules from {generated_file}...")
        try:
            gen_symbols, gen_positions = load_molecules_from_json(generated_file)
            print(f"   Loaded {len(gen_symbols)} generated molecules")
        except Exception as e:
            print(f"   ERROR loading generated molecules: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Show statistics
    print(f"\n{'4' if not generated_file else '4'}. Molecule statistics:")
    real_num_atoms = [len(s) for s in real_symbols]
    print(f"   QM9 - Average atoms: {np.mean(real_num_atoms):.2f}, Min: {min(real_num_atoms)}, Max: {max(real_num_atoms)}")
    
    real_atom_counts = {}
    for symbols in real_symbols:
        for symbol in symbols:
            real_atom_counts[symbol] = real_atom_counts.get(symbol, 0) + 1
    print(f"   QM9 - Atom distribution: {real_atom_counts}")
    
    if gen_symbols:
        gen_num_atoms = [len(s) for s in gen_symbols]
        print(f"   Generated - Average atoms: {np.mean(gen_num_atoms):.2f}, Min: {min(gen_num_atoms)}, Max: {max(gen_num_atoms)}")
        
        gen_atom_counts = {}
        for symbols in gen_symbols:
            for symbol in symbols:
                gen_atom_counts[symbol] = gen_atom_counts.get(symbol, 0) + 1
        print(f"   Generated - Atom distribution: {gen_atom_counts}")
    
    # If comparison mode, use evaluate_molecule_distributions
    results = None
    if generated_file and comparison_output:
        print(f"\n5. Running full distribution comparison (paper methodology)...")
        try:
            # Determine sample counts: use all QM9 we extracted, limit generated if specified
            if use_all_samples:
                gen_n_samples = len(gen_symbols)
                print(f"   Using ALL {gen_n_samples} generated samples (--use_all_samples flag set)")
            else:
                gen_n_samples = n_gen_samples if n_gen_samples is not None else len(gen_symbols)
            output_dir_all = os.path.join(comparison_output, 'all_samples')
            output_dir_valid = os.path.join(comparison_output, 'valid_only')

            print(f"\n   [All samples] Running report without filtering...")
            results_all = evaluate_molecule_distributions(
                real_symbols, real_positions,
                gen_symbols, gen_positions,
                output_dir=output_dir_all,
                n_real_samples=None,
                n_gen_samples=gen_n_samples,
                real_smiles=real_smiles,
                fingerprint_type='rdkit',
                random_seed=42,
                filter_invalid=False,
                ks_only=ks_only,
            )
            if not ks_only:
                print(f"   All-samples plots saved to:")
                print(f"     - {output_dir_all}/distribution_comparison_umap.png")
                print(f"     - {output_dir_all}/atom_count_distributions.png")

            print(f"\n   [Valid only] Running report with filtering...")
            results_valid = evaluate_molecule_distributions(
                real_symbols, real_positions,
                gen_symbols, gen_positions,
                output_dir=output_dir_valid,
                n_real_samples=None,
                n_gen_samples=gen_n_samples,
                real_smiles=real_smiles,
                fingerprint_type='rdkit',
                random_seed=42,
                filter_invalid=True,
                ks_only=ks_only,
            )
            if not ks_only:
                print(f"   Valid-only plots saved to:")
                print(f"     - {output_dir_valid}/distribution_comparison_umap.png")
                print(f"     - {output_dir_valid}/atom_count_distributions.png")

            results = results_valid
        except Exception as e:
            print(f"   ERROR in comparison: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Reuse results from evaluate_molecule_distributions if available (skip final plot if ks_only)
    if results is not None and ks_only:
        print("\n" + "="*60)
        print("SUCCESS: KS statistics saved (--ks_only: skipped UMAP and plots)")
        print("="*60)
        return
    if results is not None:
        # Reuse fingerprints and embeddings from evaluation results
        print(f"\n6. Reusing fingerprints and embeddings from evaluation results...")
        real_fingerprints = results['real_fingerprints']
        gen_fingerprints = results.get('generated_fingerprints', None)
        real_embedding = results['real_embedding']
        gen_embedding = results.get('generated_embedding', None)
        
        print(f"   Reused {len(real_fingerprints)} real fingerprints")
        if gen_fingerprints is not None:
            print(f"   Reused {len(gen_fingerprints)} generated fingerprints")
        print(f"   Fingerprint shape: {real_fingerprints.shape}")
        print(f"   Fingerprint sparsity: {(real_fingerprints == 0).sum() / real_fingerprints.size * 100:.2f}%")
        print(f"   Embedding shape: {real_embedding.shape}")
        print(f"   Embedding range: X=[{real_embedding[:, 0].min():.2f}, {real_embedding[:, 0].max():.2f}], "
              f"Y=[{real_embedding[:, 1].min():.2f}, {real_embedding[:, 1].max():.2f}]")
    else:
        # Compute fingerprints for QM9 (only if not already computed)
        step_num = 5 if not generated_file else 6
        print(f"\n{step_num}. Computing molecular fingerprints for QM9...")
        try:
            real_fingerprints, real_valid = compute_molecular_fingerprints(
                real_symbols, real_positions, radius=2, n_bits=2048,
                fingerprint_type='rdkit',
                filter_invalid=True
            )
            print(f"   Computed {len(real_fingerprints)} valid fingerprints")
            print(f"   Fingerprint shape: {real_fingerprints.shape}")
            print(f"   Fingerprint sparsity: {(real_fingerprints == 0).sum() / real_fingerprints.size * 100:.2f}%")
        except Exception as e:
            print(f"   ERROR computing fingerprints: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Compute fingerprints for generated if provided
        gen_fingerprints = None
        if gen_symbols:
            print(f"\n{step_num + 1}. Computing molecular fingerprints for generated molecules...")
            try:
                gen_fingerprints, gen_valid = compute_molecular_fingerprints(
                    gen_symbols, gen_positions, radius=2, n_bits=2048,
                    fingerprint_type='rdkit',
                    filter_invalid=True
                )
                print(f"   Computed {len(gen_fingerprints)} valid fingerprints")
            except Exception as e:
                print(f"   ERROR computing fingerprints: {e}")
                import traceback
                traceback.print_exc()
                return
        
        # Compute UMAP embedding
        step_num += 1 if gen_symbols else 0
        print(f"\n{step_num + 1}. Computing UMAP embedding...")
        try:
            if gen_fingerprints is not None:
                # Joint embedding for comparison
                all_fps = np.vstack([real_fingerprints, gen_fingerprints])
                all_embedding = compute_umap_embedding(
                    all_fps, n_components=2, n_neighbors=15, min_dist=0.1, random_state=42
                )
                real_embedding = all_embedding[:len(real_fingerprints)]
                gen_embedding = all_embedding[len(real_fingerprints):]
            else:
                # Single distribution
                real_embedding = compute_umap_embedding(
                    real_fingerprints, n_components=2, n_neighbors=15, min_dist=0.1, random_state=42
                )
                gen_embedding = None
            
            print(f"   Embedding shape: {real_embedding.shape}")
            print(f"   Embedding range: X=[{real_embedding[:, 0].min():.2f}, {real_embedding[:, 0].max():.2f}], "
                  f"Y=[{real_embedding[:, 1].min():.2f}, {real_embedding[:, 1].max():.2f}]")
        except Exception as e:
            print(f"   ERROR computing UMAP: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Plot distribution
    print(f"\n7. Plotting distribution...")
    try:
        if gen_embedding is not None:
            # Create side-by-side comparison plot
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
            
            # Left: QM9 only
            scatter1 = ax1.scatter(
                real_embedding[:, 0], real_embedding[:, 1], 
                alpha=0.6, s=25, 
                c=real_num_atoms[:len(real_embedding)],
                cmap='Blues', edgecolors='none'
            )
            ax1.set_xlabel('UMAP Dimension 1', fontsize=11)
            ax1.set_ylabel('UMAP Dimension 2', fontsize=11)
            ax1.set_title(f'QM9 Dataset (n={len(real_embedding)})', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            cbar1 = plt.colorbar(scatter1, ax=ax1)
            cbar1.set_label('# Atoms', fontsize=10)
            
            # Middle: Generated only
            gen_num_atoms = [len(s) for s in gen_symbols]
            scatter2 = ax2.scatter(
                gen_embedding[:, 0], gen_embedding[:, 1], 
                alpha=0.6, s=25,
                c=gen_num_atoms[:len(gen_embedding)],
                cmap='Reds', edgecolors='none'
            )
            ax2.set_xlabel('UMAP Dimension 1', fontsize=11)
            ax2.set_ylabel('UMAP Dimension 2', fontsize=11)
            ax2.set_title(f'Generated Molecules (n={len(gen_embedding)})', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            cbar2 = plt.colorbar(scatter2, ax=ax2)
            cbar2.set_label('# Atoms', fontsize=10)
            
            # Right: Overlay comparison
            ax3.scatter(real_embedding[:, 0], real_embedding[:, 1], 
                       alpha=0.4, s=20, label=f'QM9 (n={len(real_embedding)})', 
                       color='#2E86AB', edgecolors='none', marker='o')
            ax3.scatter(gen_embedding[:, 0], gen_embedding[:, 1], 
                       alpha=0.5, s=20, label=f'Generated (n={len(gen_embedding)})', 
                       color='#A23B72', edgecolors='none', marker='^')
            ax3.set_xlabel('UMAP Dimension 1', fontsize=11)
            ax3.set_ylabel('UMAP Dimension 2', fontsize=11)
            ax3.set_title('Overlay Comparison', fontsize=12, fontweight='bold')
            ax3.legend(fontsize=10, loc='best', framealpha=0.9)
            ax3.grid(True, alpha=0.3)
            
            plt.suptitle('QM9 vs Generated Distribution Comparison', fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            # Single distribution colored by number of atoms
            fig, ax = plt.subplots(figsize=(12, 10))
            scatter = ax.scatter(
                real_embedding[:, 0], 
                real_embedding[:, 1], 
                alpha=0.6, 
                s=20, 
                c=real_num_atoms[:len(real_embedding)],
                cmap='viridis',
                edgecolors='none'
            )
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Number of Atoms', fontsize=11)
            title = f'QM9 Dataset Distribution (n={len(real_embedding)} molecules)'
            ax.set_xlabel('UMAP Dimension 1', fontsize=12)
            ax.set_ylabel('UMAP Dimension 2', fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
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
    if generated_file:
        print("SUCCESS: Distribution comparison completed!")
    else:
        print("SUCCESS: QM9 distribution visualization completed!")
    print("="*60)
    print(f"\nOutput saved to: {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Test QM9 dataset distribution visualization or compare with generated molecules',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
File format for generated molecules (JSON):
{
  "molecules": [
    {
      "symbols": ["C", "H", "H", "H", "H"],
      "positions": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]
    },
    ...
  ]
}

Example usage:
  # Plot QM9 only
  python test_qm9_distribution.py --n_samples 5000
  
  # Compare generated vs QM9 (use same number for both)
  python test_qm9_distribution.py --n_samples 5000 --generated generated_molecules.json --comparison_output results/
  
  # Use more QM9 samples for stable reference, fewer generated samples
  python test_qm9_distribution.py --n_real_samples 20000 --n_gen_samples 5000 --generated generated_molecules.json --comparison_output results/
  
  # Use all generated samples (no filtering on generated)
  python test_qm9_distribution.py --use_all_samples --generated generated_molecules.json
        """
    )
    parser.add_argument('--n_samples', type=int, default=5000,
                       help='Number of samples to use for both QM9 and generated (default: 5000, overridden by --n_real_samples/--n_gen_samples)')
    parser.add_argument('--n_real_samples', type=int, default=None,
                       help='Number of QM9 samples to use (default: None = use --n_samples, or all available if not set)')
    parser.add_argument('--n_gen_samples', type=int, default=None,
                       help='Number of generated samples to use (default: None = use all available)')
    parser.add_argument('--use_all_samples', action='store_true',
                       help='Use all generated samples (overrides --n_gen_samples, but not --n_real_samples or --n_samples)')
    parser.add_argument('--output', type=str, default='qm9_distribution_test.png',
                       help='Output file path for plot (default: qm9_distribution_test.png)')
    parser.add_argument('--generated', type=str, default=None,
                       help='Path to JSON file with generated molecules (optional, enables comparison mode)')
    parser.add_argument('--comparison_output', type=str, default=None,
                       help='Directory to save full comparison results (optional, only used with --generated)')
    parser.add_argument('--ks_only', action='store_true',
                       help='Skip UMAP and plots; only compute and save KS statistics (much faster)')
    
    args = parser.parse_args()
    
    plot_qm9_distribution(
        n_samples=args.n_samples, 
        output_file=args.output,
        generated_file=args.generated,
        comparison_output=args.comparison_output,
        n_real_samples=args.n_real_samples,
        n_gen_samples=args.n_gen_samples,
        use_all_samples=args.use_all_samples,
        ks_only=args.ks_only,
    )
