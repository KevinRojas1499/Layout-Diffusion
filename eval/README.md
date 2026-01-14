# Evaluation Scripts

This directory contains scripts for evaluating generated molecule distributions against the QM9 dataset using UMAP-based visualization and molecular fingerprinting.

## Overview

The evaluation pipeline:
1. Computes molecular fingerprints (Morgan fingerprints) for both QM9 and generated molecules
2. Embeds fingerprints into 2D space using UMAP
3. Visualizes distributions side-by-side and overlaid for comparison

## Requirements

Install required dependencies:
```bash
# RDKit for molecular fingerprinting
conda install -c conda-forge rdkit

# UMAP for dimensionality reduction
pip install umap-learn
```

## Scripts

### `test_qm9_distribution.py`

Main script for visualizing QM9 distribution and comparing with generated molecules.

#### Basic Usage

**Plot QM9 dataset only:**
```bash
python eval/test_qm9_distribution.py --n_samples 5000 --output qm9_distribution.png
```

**Compare generated molecules vs QM9:**
```bash
python eval/test_qm9_distribution.py \
    --n_samples 5000 \
    --generated samples/qm9/samples.json \
    --output comparison.png \
    --comparison_output results/
```

#### Arguments

- `--n_samples` (int, default: 5000): Number of QM9 samples to use for comparison
- `--output` (str, default: `qm9_distribution_test.png`): Output file path for the plot
- `--generated` (str, optional): Path to JSON file with generated molecules (enables comparison mode)
- `--comparison_output` (str, optional): Directory to save full comparison results (only used with `--generated`)

#### Generated Molecules JSON Format

The generated molecules file should follow this format:

```json
{
  "molecules": [
    {
      "symbols": ["C", "H", "H", "H", "H"],
      "positions": [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0]
      ]
    },
    {
      "symbols": ["C", "C", "H", "H", "H", "H"],
      "positions": [
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [-0.5, 0.9, 0.0],
        [-0.5, -0.9, 0.0],
        [2.0, 0.9, 0.0],
        [2.0, -0.9, 0.0]
      ]
    }
  ]
}
```

See `example_generated_molecules.json` for a complete example.

### `evaluate_distribution.py`

Core evaluation functions (imported by `test_qm9_distribution.py`). Contains:
- `compute_molecular_fingerprints()`: Computes Morgan fingerprints from symbols/positions
- `compute_umap_embedding()`: Embeds fingerprints into 2D using UMAP
- `plot_distribution_comparison()`: Creates side-by-side and overlay visualizations
- `evaluate_molecule_distributions()`: Full evaluation pipeline
- Utility functions for loading/saving molecules from JSON

## Output Visualization

When comparing generated vs QM9, the script produces a three-panel plot:

1. **Left panel**: QM9 distribution only (colored by number of atoms, blue colormap)
2. **Middle panel**: Generated molecules only (colored by number of atoms, red colormap)
3. **Right panel**: Overlay comparison (both datasets together with distinct colors and markers)

- QM9 molecules: Blue circles (`#2E86AB`)
- Generated molecules: Purple-red triangles (`#A23B72`)

## Integration with Sampling

The `sampling.py` script in the root directory can generate molecules and save them in the required JSON format:

```bash
python sampling.py \
    --num_samples 100 \
    --num_steps 50 \
    --load_checkpoint experiments/qm9/final_checkpoint.pt \
    --dir samples/qm9
```

This will create `samples/qm9/samples.json` that can be directly used with `test_qm9_distribution.py`.

## Examples

**Quick test with 1000 QM9 samples:**
```bash
python eval/test_qm9_distribution.py --n_samples 1000
```

**Full comparison with generated molecules:**
```bash
# First, generate molecules
python sampling.py --num_samples 500 --load_checkpoint experiments/qm9/final_checkpoint.pt --dir samples/qm9

# Then compare
python eval/test_qm9_distribution.py \
    --n_samples 5000 \
    --generated samples/qm9/samples.json \
    --output qm9_vs_generated.png
```

## Notes

- The script automatically handles molecules that cannot be converted to valid RDKit structures (skips them with warnings)
- SMILES strings are preferred for accurate fingerprinting, but the script can infer bonds from 3D coordinates if SMILES are not available
- Invalid molecules (e.g., valency violations) are automatically filtered out during fingerprint computation
- The UMAP embedding uses a joint embedding when comparing distributions to ensure consistent coordinate systems
