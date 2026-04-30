"""
Batch evaluation of KS statistics at multiple generated sample sizes.

Loads QM9 once unless a matching real-side cache exists (default path under eval/cache/;
use --no-real-cache to always load QM9). With --generated, loads one JSON and evaluates each n_gen size.
With --folder, walks for .json files and evaluates each valid file (invalid files skipped).
Skips UMAP and plots. Much faster than running test_qm9_distribution.py
multiple times.

Stability mode (--n-repeats): Subsample multiple independent groups of the same
size. Outputs go under <json_stem>/n_gen_<n>/repeat-<i>/ next to each JSON file
(json_stem = filename without .json) so multiple JSONs in one folder do not clash.
"""

import argparse
import gzip
import io
import json
import os
import pickle
import sys
from contextlib import redirect_stdout
from typing import Any, Callable, Dict, Optional, Tuple

REAL_CACHE_VERSION = 1
QM9_EXTRACT_SEED = 42
QM9_MAX_LENGTH = 30
FP_RADIUS = 2
FP_BITS = 2048
FP_TYPE = 'rdkit'

DEFAULT_REAL_CACHE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'cache',
    'qm9_real_ks_bundle.pkl.gz',
)


def _real_cache_meta(args) -> Dict[str, Any]:
    return {
        'version': REAL_CACHE_VERSION,
        'n_real_samples_requested': int(args.n_real_samples),
        'extract_seed': QM9_EXTRACT_SEED,
        'qm9_max_length': QM9_MAX_LENGTH,
        'fp_radius': FP_RADIUS,
        'fp_bits': FP_BITS,
        'fp_type': FP_TYPE,
    }


def _meta_matches(stored: Dict[str, Any], expected: Dict[str, Any]) -> bool:
    for k, v in expected.items():
        if stored.get(k) != v:
            return False
    return True


def load_real_side_cache(path: str, meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        with gzip.open(path, 'rb') as f:
            payload = pickle.load(f)
    except (OSError, EOFError, pickle.UnpicklingError, gzip.BadGzipFile):
        return None
    if not isinstance(payload, dict) or 'meta' not in payload or 'bundle' not in payload:
        return None
    if not _meta_matches(payload['meta'], meta):
        return None
    bundle = payload['bundle']
    need = ('real_symbols', 'real_positions', 'real_smiles', 'precomputed_real_all', 'precomputed_real_valid')
    if any(k not in bundle for k in need):
        return None
    if payload['meta'].get('n_molecules') != len(bundle['real_symbols']):
        return None
    return bundle


def save_real_side_cache(path: str, meta: Dict[str, Any], bundle: Dict[str, Any]) -> None:
    meta = {**meta, 'n_molecules': len(bundle['real_symbols'])}
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + '.tmp'
    with gzip.open(tmp, 'wb', compresslevel=3) as f:
        pickle.dump({'meta': meta, 'bundle': bundle}, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)

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


def iter_json_files(root: str):
    root = os.path.abspath(root)
    for dirpath, _, filenames in os.walk(root):
        for name in sorted(filenames):
            if name.lower().endswith('.json'):
                yield os.path.join(dirpath, name)


def eval_output_dir_next_to_json(
    source_dir: str, json_stem: str, n_gen: int, repeat: int, n_repeats: int
) -> str:
    """Directory for one KS run: <source_dir>/<stem>/n_gen_<n>/[repeat-<r>/]."""
    stem = json_stem or 'samples'
    base = os.path.join(source_dir, stem, f'n_gen_{n_gen}')
    if n_repeats > 1:
        return os.path.join(base, f'repeat-{repeat}')
    return base


def try_load_valid_molecules_json(path: str) -> Optional[Tuple[list, list]]:
    """Load molecules if the file is valid QM9-style JSON; otherwise return None."""
    try:
        with redirect_stdout(io.StringIO()):
            symbols, positions = load_molecules_from_json(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, KeyError):
        return None
    if not symbols:
        return None
    return symbols, positions


def run_eval_loop(
    args,
    real_symbols,
    real_positions,
    real_smiles,
    precomputed_real_all,
    precomputed_real_valid,
    gen_symbols,
    gen_positions,
    get_eval_output_dir: Callable[[int, int], str],
    label: str = '',
):
    n_repeats = max(1, args.n_repeats)
    for n_gen in args.n_gen_samples:
        for repeat in range(n_repeats):
            output_dir = get_eval_output_dir(n_gen, repeat)

            valid_dir = os.path.join(output_dir, 'valid_only')
            all_dir = os.path.join(output_dir, 'all_samples')
            summary_path = os.path.join(valid_dir, 'ks_statistics_summary.txt')

            if args.skip_existing and os.path.exists(summary_path):
                if n_repeats > 1:
                    print(f"\nSkipping {label}n_gen={n_gen} repeat={repeat} (exists)")
                else:
                    print(f"\nSkipping {label}n_gen={n_gen} (exists)")
                continue

            seed = 42 + repeat
            if n_repeats > 1:
                print(f"\n{'='*50}")
                print(f"{label}Evaluating n_gen_samples={n_gen} repeat={repeat}/{n_repeats} (seed={seed})")
                print(f"{'='*50}")
            else:
                print(f"\n{'='*50}")
                print(f"{label}Evaluating n_gen_samples={n_gen}")
                print(f"{'='*50}")

            evaluate_molecule_distributions(
                real_symbols, real_positions,
                gen_symbols, gen_positions,
                output_dir=all_dir,
                n_real_samples=None,
                n_gen_samples=n_gen,
                real_smiles=real_smiles,
                fingerprint_type='rdkit',
                random_seed=seed,
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
                random_seed=seed,
                filter_invalid=True,
                ks_only=True,
                precomputed_real=precomputed_real_valid,
            )
            print(f"   Saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description='Batch KS evaluation at multiple generated sample sizes (loads data once)',
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--generated', type=str,
                     help='Path to samples.json')
    src.add_argument('--folder', type=str,
                     help='Root directory: recursively find .json files; each valid file is evaluated '
                          'and results go under <same-dir>/<stem>/n_gen_<n>/valid_only|all_samples '
                          '(stem = filename without .json). Invalid JSON files are skipped silently.')
    parser.add_argument('--n-gen-samples', type=int, nargs='+', required=True,
                        help='Generated sample sizes to evaluate (e.g. 1000 2500 5000 10000)')
    parser.add_argument('--n-real-samples', type=int, default=132008,
                        help='QM9 samples (default: 132008 = all)')
    parser.add_argument('--n-repeats', type=int, default=1,
                        help='Number of independent subsamples per size (for stability analysis). '
                             'Each repeat uses a different random subsample. Default 1.')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                        help='Skip sizes where output exists')
    parser.add_argument('--no-skip-existing', action='store_false', dest='skip_existing',
                        help='Do not skip existing')
    parser.add_argument('--real-cache', type=str, default=DEFAULT_REAL_CACHE, metavar='PATH',
                        help='Path to a .pkl.gz: load QM9 real extract + precomputed fingerprints/properties '
                             'if metadata matches; otherwise compute and save. Default: %(default)s')
    parser.add_argument('--no-real-cache', action='store_true',
                        help='Disable real-side cache (always load QM9 and precompute; do not read or write cache)')
    args = parser.parse_args()
    if args.no_real_cache:
        args.real_cache = None

    print("="*60)
    print("Batch KS evaluation (load once, evaluate multiple sizes)")
    print("="*60)

    cache_meta = _real_cache_meta(args)
    bundle_from_cache = None
    if args.real_cache:
        print(f"\n0. Real-side cache: {args.real_cache!r}")
        bundle_from_cache = load_real_side_cache(args.real_cache, cache_meta)
        if bundle_from_cache is not None:
            print("   Loaded valid cache; skipping QM9 load and real precompute.")
        else:
            print("   No valid cache (missing or stale); will compute and write.")

    if bundle_from_cache is not None:
        real_symbols = bundle_from_cache['real_symbols']
        real_positions = bundle_from_cache['real_positions']
        real_smiles = bundle_from_cache['real_smiles']
        precomputed_real_all = bundle_from_cache['precomputed_real_all']
        precomputed_real_valid = bundle_from_cache['precomputed_real_valid']
    else:
        # 1. Load QM9 once
        print("\n1. Loading QM9...")
        tokenizer = VocabTokenizer(vocab={'H', 'C', 'N', 'O', 'F'})
        dataset = QM9Dataset(tokenizer, max_length=QM9_MAX_LENGTH)
        n_real = min(args.n_real_samples, len(dataset))
        real_symbols, real_positions, real_smiles = extract_molecules_from_qm9_dataset(
            dataset, n_samples=n_real, random_seed=QM9_EXTRACT_SEED
        )
        print(f"   Loaded {len(real_symbols)} real molecules")

        # 2. Precompute real molecules data once (reused for all n_gen sizes)
        print("\n2. Precomputing real molecules (fingerprints + properties)...")
        print("   [all_samples mode - no filtering]")
        real_props_all = compute_molecular_properties(real_symbols, real_positions, filter_invalid=False)
        real_fps_all, real_valid_all, real_stats_all = compute_molecular_fingerprints(
            real_symbols, real_positions,
            radius=FP_RADIUS, n_bits=FP_BITS,
            fingerprint_type=FP_TYPE,
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
            radius=FP_RADIUS, n_bits=FP_BITS,
            fingerprint_type=FP_TYPE,
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

        if args.real_cache:
            save_bundle = {
                'real_symbols': real_symbols,
                'real_positions': real_positions,
                'real_smiles': real_smiles,
                'precomputed_real_all': precomputed_real_all,
                'precomputed_real_valid': precomputed_real_valid,
            }
            save_real_side_cache(args.real_cache, cache_meta, save_bundle)
            print(f"   Saved real-side cache to {args.real_cache}")

    n_repeats = max(1, args.n_repeats)

    if args.folder:
        json_paths = list(iter_json_files(args.folder))
        print(f"\n3. Folder mode: found {len(json_paths)} JSON file(s) under {args.folder!r}")
        for json_path in json_paths:
            loaded = try_load_valid_molecules_json(json_path)
            if loaded is None:
                continue
            gen_symbols, gen_positions = loaded
            json_dir = os.path.dirname(os.path.abspath(json_path))
            json_stem = os.path.splitext(os.path.basename(json_path))[0]
            rel = os.path.relpath(json_path, start=os.path.abspath(args.folder))

            def get_eval_output_dir(n_gen: int, repeat: int, _jd=json_dir, _stem=json_stem) -> str:
                return eval_output_dir_next_to_json(_jd, _stem, n_gen, repeat, n_repeats)

            print(f"\n--- Valid: {rel} ({len(gen_symbols)} molecules) ---")
            run_eval_loop(
                args,
                real_symbols, real_positions, real_smiles,
                precomputed_real_all, precomputed_real_valid,
                gen_symbols, gen_positions,
                get_eval_output_dir,
                label=f'[{rel}] ',
            )
    else:
        gen_stem = os.path.splitext(os.path.basename(args.generated))[0]

        def get_eval_output_dir_single(n_gen: int, repeat: int) -> str:
            source_dir = os.path.dirname(os.path.abspath(args.generated))
            return eval_output_dir_next_to_json(source_dir, gen_stem, n_gen, repeat, n_repeats)

        print("\n3. Loading generated molecules...")
        gen_symbols, gen_positions = load_molecules_from_json(args.generated)
        print(f"   Loaded {len(gen_symbols)} generated molecules")

        run_eval_loop(
            args,
            real_symbols, real_positions, real_smiles,
            precomputed_real_all, precomputed_real_valid,
            gen_symbols, gen_positions,
            get_eval_output_dir_single,
            label='',
        )

    print("\n" + "="*60)
    print("Done")
    print("="*60)


if __name__ == '__main__':
    main()
