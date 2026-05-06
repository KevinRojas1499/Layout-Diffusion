"""Quick sampler for the pre-fuse-layer MMDiTQM9 checkpoint at
experiments/qm9-matching-cluster/itr_1350000/snapshot.pt.

Uses models/mmdit_qm9_legacy.py (a copy of mmdit_qm9.py at git rev dad71ef,
which exactly matches the checkpoint's state_dict).

Single-GPU; writes samples.json to --dir.
"""
import os, json, click, torch
from tqdm import tqdm
from multimodal_interpolant import MultimodalInterpolant
from custom_datasets.qm9 import QM9Dataset
from utils.tokenizer import VocabTokenizer
from models.mmdit_qm9_legacy import MMDiTQM9


@click.command()
@click.option('--load_checkpoint', default='experiments/qm9-matching-cluster/itr_1350000/snapshot.pt')
@click.option('--dir', 'out_dir', default='samples-legacy-test')
@click.option('--num_samples', type=int, default=50)
@click.option('--num_steps', type=int, default=100)
@click.option('--batch_size', type=int, default=50)
@click.option('--sampler', type=click.Choice(['euler', 'split', 'staggered']), default='staggered')
@click.option('--use_ema', is_flag=True, default=False)
@click.option('--seed', type=int, default=42)
def main(load_checkpoint, out_dir, num_samples, num_steps, batch_size, sampler, use_ema, seed):
    torch.manual_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tok = VocabTokenizer(vocab={'H', 'C', 'N', 'O', 'F'})
    dataset = QM9Dataset(tok)

    model = MMDiTQM9(
        euclidean_dim=3,
        vocab_size=tok.vocab_size,
        symbols_depth=6,
        positions_depth=6,
        depth=6,
        dim_modalities=[384, 384],
        dim_joint_attn=384,
        dim_conds=[384, 384],
    ).to(device)

    snap = torch.load(load_checkpoint, weights_only=True, map_location=device)
    key = 'ema' if use_ema else 'model'
    model.load_state_dict(snap[key], strict=True)
    model.eval()
    print(f"Loaded {key} from {load_checkpoint}")
    print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    interp = MultimodalInterpolant(
        max_length=dataset.max_length,
        vocab_size=tok.vocab_size,
        mask_token=tok.mask_token_id,
        pad_token=tok.pad_token_id,
        bos_token=tok.bos_token_id,
        euclidean_dim=3,
    )

    os.makedirs(out_dir, exist_ok=True)
    output = {'molecules': []}
    n_batches = (num_samples + batch_size - 1) // batch_size
    for _ in tqdm(range(n_batches), desc='Sampling'):
        samples = interp.sampling(
            model, num_steps, batch_size, dataset.max_length, device,
            return_trace=False, sampler=sampler,
        )
        for s in samples:
            symbols = tok.decode(s.yt.cpu())
            positions = s.xt.cpu()[1:len(symbols) + 1, :]
            output['molecules'].append({'symbols': list(symbols), 'positions': positions.tolist()})

    with open(os.path.join(out_dir, 'samples.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {len(output['molecules'])} molecules to {out_dir}/samples.json")


if __name__ == '__main__':
    main()
