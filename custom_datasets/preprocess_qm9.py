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


def preprocess_qm9_dataset(
    output_dir: str,
):
    """
    Preprocess QM9 dataset by filtering to molecules that pass the same
    chemistry checks used during evaluation (OpenBabel -> RDKit validity and
    connectedness).
    
    Args:
        output_dir: Directory to save the preprocessed dataset
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
            
            result = {
                'atomic_symbols': atomic_symbols,
                'pos': pos,
                'canonical_smiles': smiles,
                # Keep compatibility with previous training codepaths.
                'original_atomic_symbols': atomic_symbols,
                'original_pos': pos,
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
    args = parser.parse_args()
    
    preprocess_qm9_dataset(
        output_dir=args.output_dir,
    )
