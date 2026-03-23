"""
Batch evaluation of KS statistics at multiple generated sample sizes.

Loads QM9 and generated molecules once, then evaluates each n_gen size.
Skips UMAP and plots. Much faster than running test_qm9_distribution.py
multiple times.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from custom_datasets.qm9 import QM9Dataset
from utils.tokenizer import VocabTokenizer
from eval.evaluate_distribution import (
    extract_molecules_from_qm9_dataset,
    load_molecules_from_json,
    compute_molecular_properties,
    compute_molecular_fingerprints,
    evaluate_molecule_distributions,
)


def main():
    parser = argparse.ArgumentParser(
        description='Batch KS evaluation at multiple generated sample sizes (loads data once)',
    )
    parser.add_argument('--generated', type=str, required=True,
                        help='Path to samples.json')
    parser.add_argument('--n-gen-samples', type=int, nargs='+', required=True,
                        help='Generated sample sizes to evaluate (e.g. 1000 2500 5000 10000)')
    parser.add_argument('--n-real-samples', type=int, default=132008,
                        help='QM9 samples (default: 132008 = all)')
    parser.add_argument('--comparison-output-template', type=str, default='results/eval-n-gen-{n_gen}',
                        help='Output dir template, use {n_gen} for size')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                        help='Skip sizes where output exists')
    parser.add_argument('--no-skip-existing', action='store_false', dest='skip_existing',
                        help='Do not skip existing')
    args = parser.parse_args()

    print("="*60)
    print("Batch KS evaluation (load once, evaluate multiple sizes)")
    print("="*60)

    # 1. Load QM9 once
    print("\n1. Loading QM9...")
    tokenizer = VocabTokenizer(vocab={'H', 'C', 'N', 'O', 'F'})
    dataset = QM9Dataset(tokenizer, max_length=30)
    n_real = min(args.n_real_samples, len(dataset))
    real_symbols, real_positions, real_smiles = extract_molecules_from_qm9_dataset(
        dataset, n_samples=n_real, random_seed=42
    )
    print(f"   Loaded {len(real_symbols)} real molecules")

    # 2. Load generated once
    print("\n2. Loading generated molecules...")
    gen_symbols, gen_positions = load_molecules_from_json(args.generated)
    print(f"   Loaded {len(gen_symbols)} generated molecules")

    # 3. Precompute real molecules data once (reused for all n_gen sizes)
    print("\n3. Precomputing real molecules (fingerprints + properties)...")
    print("   [all_samples mode - no filtering]")
    real_props_all = compute_molecular_properties(real_symbols, real_positions, filter_invalid=False)
    real_fps_all, real_valid_all, real_stats_all = compute_molecular_fingerprints(
        real_symbols, real_positions,
        radius=2, n_bits=2048,
        fingerprint_type='rdkit',
        filter_invalid=False,
        return_stats=True
    )
    precomputed_real_all = {
        'properties': real_props_all,
        'fps': real_fps_all,
        'valid': real_valid_all,
        'fp_stats': real_stats_all,
    }
    print("   [valid_only mode - filter invalid]")
    real_props_valid = compute_molecular_properties(real_symbols, real_positions, filter_invalid=True)
    real_fps_valid, real_valid_valid, real_stats_valid = compute_molecular_fingerprints(
        real_symbols, real_positions,
        radius=2, n_bits=2048,
        fingerprint_type='rdkit',
        filter_invalid=True,
        return_stats=True
    )
    precomputed_real_valid = {
        'properties': real_props_valid,
        'fps': real_fps_valid,
        'valid': real_valid_valid,
        'fp_stats': real_stats_valid,
    }
    print("   Done. Real data will be reused for all n_gen sizes.")

    for n_gen in args.n_gen_samples:
        output_dir = args.comparison_output_template.format(n_gen=n_gen)
        valid_dir = os.path.join(output_dir, 'valid_only')
        all_dir = os.path.join(output_dir, 'all_samples')
        summary_path = os.path.join(valid_dir, 'ks_statistics_summary.txt')

        if args.skip_existing and os.path.exists(summary_path):
            print(f"\nSkipping n_gen={n_gen} (exists)")
            continue

        print(f"\n{'='*50}")
        print(f"Evaluating n_gen_samples={n_gen}")
        print(f"{'='*50}")

        evaluate_molecule_distributions(
            real_symbols, real_positions,
            gen_symbols, gen_positions,
            output_dir=all_dir,
            n_real_samples=None,
            n_gen_samples=n_gen,
            real_smiles=real_smiles,
            fingerprint_type='rdkit',
            random_seed=42,
            filter_invalid=False,
            ks_only=True,
            precomputed_real=precomputed_real_all,
        )

        evaluate_molecule_distributions(
            real_symbols, real_positions,
            gen_symbols, gen_positions,
            output_dir=valid_dir,
            n_real_samples=None,
            n_gen_samples=n_gen,
            real_smiles=real_smiles,
            fingerprint_type='rdkit',
            random_seed=42,
            filter_invalid=True,
            ks_only=True,
            precomputed_real=precomputed_real_valid,
        )
        print(f"   Saved to {output_dir}/")

    print("\n" + "="*60)
    print("Done")
    print("="*60)


if __name__ == '__main__':
    main()
