"""
Preprocessing script for QM9 dataset.
Builds a filtered train set (valid + fully connected molecules) and
saves it for faster loading during training/evaluation.
"""
import datasets
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
from eval.evaluate_distribution import _positions_to_molecule_via_openbabel
from custom_datasets.qm9_hydrogen_order import reorder_hydrogens_nearest_heavy


def preprocess_qm9_dataset(
    output_dir: str,
    reorder_hydrogens: bool = True,
):
    """
    Preprocess QM9 dataset by filtering to molecules that pass the same
    chemistry checks used during evaluation (OpenBabel -> RDKit validity and
    connectedness).
    
    Args:
        output_dir: Directory to save the preprocessed dataset
        reorder_hydrogens: If True, place each H before its nearest heavy atom
            (heavy order unchanged); original order is kept in original_* keys.
    """
    print("Loading QM9 dataset...")
    qm9_dataset = datasets.load_dataset('yairschiff/qm9', split='train')
    print(f"Processing {len(qm9_dataset)} molecules...")
    
    def process_example(example):
        """Process a single example and keep only valid, connected molecules."""
        atomic_symbols = example['atomic_symbols']
        pos = example['pos']
        smiles = example.get('canonical_smiles') or example.get('smiles')
        
        try:
            pos_np = np.array(pos)
            _, _, is_valid, is_fully_connected = _positions_to_molecule_via_openbabel(
                atomic_symbols, pos_np
            )
            if not is_valid or not is_fully_connected:
                return None

            orig_symbols = list(atomic_symbols)
            orig_pos = np.asarray(pos, dtype=np.float64)
            if reorder_hydrogens:
                new_symbols, new_pos = reorder_hydrogens_nearest_heavy(
                    orig_symbols, orig_pos
                )
                out_symbols = new_symbols
                out_pos = new_pos.tolist()
            else:
                out_symbols = orig_symbols
                out_pos = pos if isinstance(pos, list) else orig_pos.tolist()

            result = {
                'atomic_symbols': out_symbols,
                'pos': out_pos,
                'canonical_smiles': smiles,
                'original_atomic_symbols': orig_symbols,
                'original_pos': orig_pos.tolist(),
            }
            
            for key in example.keys():
                if key not in ['atomic_symbols', 'pos', 'canonical_smiles', 'smiles']:
                    result[key] = example[key]
            
            return result
            
        except Exception:
            return None
    
    processed_data = []
    filtered_out_count = 0
    print("Filtering molecules (valid + fully connected)...")
    for example in tqdm(qm9_dataset, desc="Processing"):
        result = process_example(example)
        if result is not None:
            processed_data.append(result)
        else:
            filtered_out_count += 1
    
    print(f"Successfully processed {len(processed_data)} molecules")
    print(f"Filtered out {filtered_out_count} molecules")
    
    # Create new dataset from processed data
    print("Creating preprocessed dataset...")
    processed_dataset = datasets.Dataset.from_list(processed_data)
    
    # Save dataset
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving preprocessed dataset to {output_path}...")
    processed_dataset.save_to_disk(str(output_path))
    
    print(f"Preprocessing complete! Dataset saved to {output_path}")
    print(f"Dataset size: {len(processed_dataset)} molecules")
    
    return processed_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and filter QM9 dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the preprocessed dataset")
    parser.add_argument(
        "--no_hydrogen_reorder",
        action="store_true",
        help="Keep raw dataset atom order (heavy/H as in QM9 files).",
    )
    args = parser.parse_args()

    preprocess_qm9_dataset(
        output_dir=args.output_dir,
        reorder_hydrogens=not args.no_hydrogen_reorder,
    )
