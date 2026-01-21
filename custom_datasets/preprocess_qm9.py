"""
Preprocessing script for QM9 dataset.
Reorders atoms using the correct SMILES-based ordering from qm_utils.py
and saves the preprocessed dataset for faster loading.
"""
import datasets
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
from custom_datasets.qm_utils import reorder_like_branching_flows


def preprocess_qm9_dataset(
    output_dir: str,
    charge: int = 0,
):
    """
    Preprocess QM9 dataset by reordering atoms according to canonical SMILES order.
    
    Args:
        output_dir: Directory to save the preprocessed dataset
        charge: Molecular charge for bond determination (default 0)
    """
    print("Loading QM9 dataset...")
    qm9_dataset = datasets.load_dataset('yairschiff/qm9', split='train')
    print(f"Processing {len(qm9_dataset)} molecules...")
    
    def process_example(example):
        """Process a single example, reordering atoms."""
        atomic_symbols = example['atomic_symbols']
        pos = example['pos']
        smiles = example.get('canonical_smiles') or example.get('smiles')
        
        if smiles is None:
            # Skip if no SMILES available
            return None
        
        try:
            # Reorder atoms using the correct implementation
            new_symbols, new_pos = reorder_like_branching_flows(
                atomic_symbols, pos, smiles, charge=charge
            )
            
            # Convert numpy arrays to lists for serialization
            new_pos = new_pos.tolist() if isinstance(new_pos, np.ndarray) else new_pos
            
            # Return preprocessed data
            result = {
                'atomic_symbols': new_symbols,
                'pos': new_pos,
                'canonical_smiles': smiles,
                # Keep original data for reference
                'original_atomic_symbols': atomic_symbols,
                'original_pos': pos,
            }
            
            # Copy over any other fields from the original example
            for key in example.keys():
                if key not in ['atomic_symbols', 'pos', 'canonical_smiles', 'smiles']:
                    result[key] = example[key]
            
            return result
            
        except Exception as e:
            # Log error but continue processing
            # print(f"Error processing molecule with SMILES {smiles}: {e}")
            return None
    
    # Process all examples
    processed_data = []
    failed_count = 0
    print("Reordering atoms...")
    for i, example in enumerate(tqdm(qm9_dataset, desc="Processing")):
        result = process_example(example)
        if result is not None:
            processed_data.append(result)
        else:
            failed_count += 1
    
    print(f"Successfully processed {len(processed_data)} molecules")
    if failed_count > 0:
        print(f"Failed to process {failed_count} molecules")
    
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
    parser = argparse.ArgumentParser(description="Preprocess QM9 dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the preprocessed dataset"
    )
    parser.add_argument(
        "--charge",
        type=int,
        default=0,
        help="Molecular charge for bond determination"
    )
    
    args = parser.parse_args()
    
    preprocess_qm9_dataset(
        output_dir=args.output_dir,
        charge=args.charge,
    )
