"""Training script for the fixed-length multimodal interpolant baseline.

This is an oracle baseline: at sampling time the sequence length is drawn from
the empirical training distribution (perfect length model), so the model only
has to learn content given a known length.
"""
import json
import os
import random
import click
import numpy as np
import torch
from collections import OrderedDict, defaultdict
from copy import deepcopy
from torch.utils.data import DataLoader
from tqdm import tqdm
from fixed_length_multimodal_interpolant import (
    FixedLengthMultimodalInterpolant,
    SamplingResult,
)
from custom_datasets.multimodal_math import ParenthesizedEquationsDataset
from utils.misc import dotdict
from utils.tokenizer import VocabTokenizer
from utils.optimizers import WarmUpScheduler
from models.mmdit_qm9 import MMDiTFixedLength
from evaluate_equations_parenthesis import parse_equation


# ---------------------------------------------------------------------------
# W&B helper
# ---------------------------------------------------------------------------

def init_wandb(opts):
    import wandb
    wandb.init(
        project='MMDiT-Toy',
        name=f'toy-{opts.run_name}',
        tags=['training', 'fixed-length'],
        config=opts,
    )


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_samples(samples, character_tokenizer, euclidean_dim=1, delta=0.5, num_scale=1.0):
    """Compute accuracy metrics on a SamplingResult batch."""
    n = samples.xt.shape[0]
    valid = 0
    correct = 0
    invalid = 0
    diffs = []

    for i in range(n):
        symbols_str = character_tokenizer.decode(samples.yt[i].cpu())
        symbols = list(symbols_str)
        positions = samples.xt[i].cpu()[1:len(symbols) + 1, :] * num_scale
        numbers = positions.squeeze(-1).tolist() if euclidean_dim == 1 else positions.tolist()

        try:
            left, right = parse_equation(symbols, numbers)
            valid += 1
            diff = abs(left - right)
            diffs.append(diff)
            if diff < delta:
                correct += 1
        except Exception:
            invalid += 1

    accuracy = correct / valid if valid > 0 else 0.0
    mean_error = float(np.mean(diffs)) if diffs else float('inf')

    return {
        'total': n,
        'valid': valid,
        'invalid': invalid,
        'correct': correct,
        'accuracy': accuracy,
        'mean_abs_error': mean_error,
    }


def save_samples_jsonl(samples, character_tokenizer, path, euclidean_dim=1, num_scale=1.0):
    """Write generated samples to a JSONL file."""
    with open(path, 'w') as f:
        for i in range(samples.xt.shape[0]):
            symbols_str = character_tokenizer.decode(samples.yt[i].cpu())
            symbols = list(symbols_str)
            positions = samples.xt[i].cpu()[1:len(symbols) + 1, :] * num_scale
            numbers = positions.squeeze(-1).tolist() if euclidean_dim == 1 else positions.tolist()
            f.write(json.dumps({'symbols': symbols, 'numbers': numbers, 'length': len(symbols)}) + '\n')


def collect_length_pairs(dataset) -> list:
    """Return ordered list of (x_length, y_length) pairs from a dataset."""
    return [(len(item['numbers']), len(item['symbols'])) for item in dataset.data]


@torch.no_grad()
def oracle_sampling(
    interpolant: FixedLengthMultimodalInterpolant,
    model: torch.nn.Module,
    steps: int,
    eval_samples: int,
    length_pairs: list,
    device: torch.device,
) -> SamplingResult:
    """Generate eval_samples using lengths drawn in order from the eval dataset.

    Each generated sample gets the length of the corresponding eval dataset entry
    (cycling through the list if eval_samples > len(length_pairs)).  This is a
    strict oracle: no random resampling — the exact test-set length distribution
    is replicated sample-for-sample.
    """
    max_length = interpolant.max_length
    full_len = max_length + 1  # includes BOS

    # Use test lengths in order, cycling if eval_samples > dataset size.
    sampled_lengths = [length_pairs[i % len(length_pairs)] for i in range(eval_samples)]

    # Group indices by length pair to enable batched sampling.
    groups = defaultdict(list)
    for i, pair in enumerate(sampled_lengths):
        groups[pair].append(i)

    all_xt = torch.zeros(eval_samples, full_len, interpolant.euclidean_dim, device=device)
    all_yt = torch.full((eval_samples, full_len), interpolant.pad_token, dtype=torch.long, device=device)
    all_x_mask = torch.zeros(eval_samples, full_len, dtype=torch.bool, device=device)
    all_y_mask = torch.zeros(eval_samples, full_len, dtype=torch.bool, device=device)

    for (x_len, y_len), indices in tqdm(groups.items(), desc='oracle lengths', leave=False):
        result = interpolant.sampling(
            model=model,
            steps=steps,
            batch_size=len(indices),
            x_length=x_len,
            y_length=y_len,
            max_length=max_length,
            device=device,
        )
        for j, orig_idx in enumerate(indices):
            all_xt[orig_idx] = result.xt[j]
            all_yt[orig_idx] = result.yt[j]
            all_x_mask[orig_idx] = result.x_mask_t[j]
            all_y_mask[orig_idx] = result.y_mask_t[j]

    return SamplingResult(
        xt=all_xt,
        yt=all_yt,
        x_mask_t=all_x_mask,
        y_mask_t=all_y_mask,
        trajectory=[],
    )


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_ckpt(model, ema, opt, scheduler, path):
    torch.save({
        'model': model.state_dict(),
        'ema': ema.state_dict(),
        'optimizer': opt.state_dict(),
        'scheduler': scheduler.state_dict(),
    }, path)


def load_checkpoint(opts, device, model, ema, opt, scheduler):
    snapshot = torch.load(opts.load_checkpoint, weights_only=True, map_location=device)
    model.load_state_dict(snapshot['model'], strict=False)
    ema.load_state_dict(snapshot['ema'], strict=False)
    opt.load_state_dict(snapshot['optimizer'])
    scheduler.load_state_dict(snapshot['scheduler'])
    last = snapshot['scheduler'].get('last_epoch', None)
    return last if last is not None else scheduler.last_epoch


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@click.command()
@click.option('--data_path', type=str, default=None)
@click.option('--eval_data_path', type=str, default=None,
              help='Dataset to draw evaluation lengths from (defaults to --data_path)')
@click.option('--optimizer', type=click.Choice(['adam', 'adamw']), default='adamw')
@click.option('--ema_beta', type=float, default=0.9999)
@click.option('--lr', type=float, default=1e-4)
@click.option('--batch_size', type=int, default=128)
@click.option('--log_rate', type=int, default=500)
@click.option('--num_iters', type=int, default=5000)
@click.option('--warmup_iters', type=int, default=500)
@click.option('--num_workers', type=int, default=2)
@click.option('--seed', type=int, default=42)
@click.option('--dir', type=str)
@click.option('--load_checkpoint', type=str, default=None,
              help='Path to a snapshot.pt to resume from')
@click.option('--finetune_checkpoint', type=str, default=None,
              help='Load model weights only (fresh optimizer/scheduler)')
@click.option('--enable_wandb', is_flag=True, default=False)
@click.option('--run_name', type=str, default='')
@click.option('--eval_samples', type=int, default=200)
@click.option('--eval_steps', type=int, default=50)
@click.option('--eval_delta', type=float, default=0.5)
@click.option('--lr_schedule', type=click.Choice(['warmup_only', 'cosine']), default='warmup_only')
@click.option('--lr_min', type=float, default=1e-6)
@click.option('--num_scale', type=float, default=1.0)
@click.option('--depth', type=int, default=4)
@click.option('--hidden_dim', type=int, default=256)
def training(**opts):
    opts = dotdict(opts)
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed(opts.seed)
    torch.cuda.manual_seed_all(opts.seed)
    random.seed(opts.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    euclidean_dim = 1
    hidden_dim = opts.hidden_dim
    character_tokenizer = VocabTokenizer(vocab={'+', '-', '*', '=', '(', ')'})
    dataset = ParenthesizedEquationsDataset(character_tokenizer, data_path=opts.data_path)

    eval_data_path = opts.eval_data_path if opts.eval_data_path else opts.data_path
    if eval_data_path != opts.data_path:
        eval_dataset = ParenthesizedEquationsDataset(character_tokenizer, data_path=eval_data_path)
    else:
        eval_dataset = dataset
    eval_length_pairs = collect_length_pairs(eval_dataset)

    print('Vocab')
    print('--------------------------------')
    for token, idx in character_tokenizer.atom_to_idx.items():
        print(f'{token}: {idx}')
    print('--------------------------------')
    print(f'Train dataset max_length: {dataset.max_length}  ({len(dataset)} samples)')
    print(f'Eval dataset: {eval_data_path}  ({len(eval_dataset)} samples, {len(set(eval_length_pairs))} unique length pairs)')

    dataloader = DataLoader(
        dataset, batch_size=opts.batch_size, shuffle=True,
        num_workers=opts.num_workers, drop_last=True,
    )

    if opts.enable_wandb:
        init_wandb(opts)

    model = MMDiTFixedLength(
        euclidean_dim=euclidean_dim,
        vocab_size=character_tokenizer.vocab_size,
        symbols_depth=opts.depth,
        positions_depth=opts.depth,
        depth=opts.depth,
        dim_modalities=[hidden_dim, hidden_dim],
        dim_joint_attn=hidden_dim,
        dim_conds=[hidden_dim, hidden_dim],
    ).to(device)
    ema = deepcopy(model)

    opt = torch.optim.AdamW(model.parameters(), lr=opts.lr)
    if opts.lr_schedule == 'cosine':
        warmup = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=1e-8, end_factor=1.0, total_iters=opts.warmup_iters
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=opts.num_iters - opts.warmup_iters, eta_min=opts.lr_min
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup, cosine], milestones=[opts.warmup_iters]
        )
    else:
        scheduler = WarmUpScheduler(opt, opts.warmup_iters)
    scaler = torch.amp.GradScaler()

    interpolant = FixedLengthMultimodalInterpolant(
        max_length=dataset.max_length,
        vocab_size=character_tokenizer.vocab_size,
        mask_token=character_tokenizer.mask_token_id,
        pad_token=character_tokenizer.pad_token_id,
        bos_token=character_tokenizer.bos_token_id,
        euclidean_dim=euclidean_dim,
    )

    start_iter = 0
    if opts.load_checkpoint is not None:
        start_iter = load_checkpoint(opts, device, model, ema, opt, scheduler)
    elif opts.finetune_checkpoint is not None:
        snap = torch.load(opts.finetune_checkpoint, weights_only=True, map_location=device)
        model.load_state_dict(snap['model'], strict=False)
        ema.load_state_dict(snap['ema'], strict=False)
        print(f'Loaded weights from {opts.finetune_checkpoint} (fresh optimizer)')

    model.train()
    os.makedirs(opts.dir, exist_ok=True)
    metrics_path = os.path.join(opts.dir, 'metrics.jsonl')

    training_iter = start_iter
    while training_iter < opts.num_iters:
        pbar = tqdm(dataloader, total=len(dataloader), leave=False)
        for data_ in pbar:
            if training_iter >= opts.num_iters:
                break

            for key, value in data_.items():
                data_[key] = value.to(device=device)
            if opts.num_scale != 1.0:
                data_['x'] = data_['x'] / opts.num_scale

            opt.zero_grad()
            with torch.autocast('cuda', dtype=torch.float16):
                losses = interpolant.compute_loss(model, data_)

            loss = losses['dsm_loss'] + losses['discrete_unmasking_loss']

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            update_ema(ema, model, decay=opts.ema_beta)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            for param in model.parameters():
                if param.grad is not None:
                    torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)
            scaler.step(opt)
            scaler.update()
            scheduler.step()

            training_iter += 1

            pbar.set_description(
                f'Iter {training_iter} --- '
                f'DSM: {losses["dsm_loss"]:.4f}  '
                f'Disc: {losses["discrete_unmasking_loss"]:.4f}'
            )

            if opts.enable_wandb:
                import wandb
                log_dict = {'loss': loss.detach().item(), 'step': training_iter}
                log_dict.update({k: v.item() for k, v in losses.items()})
                wandb.log(log_dict)

            if training_iter % opts.log_rate == 0 or training_iter == opts.num_iters:
                ckpt_path = os.path.join(opts.dir, f'itr_{training_iter}')
                os.makedirs(ckpt_path, exist_ok=True)
                save_ckpt(model, ema, opt, scheduler, os.path.join(ckpt_path, 'snapshot.pt'))

                ema.eval()
                with torch.no_grad():
                    samples = oracle_sampling(
                        interpolant=interpolant,
                        model=ema,
                        steps=opts.eval_steps,
                        eval_samples=opts.eval_samples,
                        length_pairs=eval_length_pairs,
                        device=device,
                    )

                save_samples_jsonl(
                    samples, character_tokenizer,
                    os.path.join(ckpt_path, 'samples.jsonl'),
                    euclidean_dim=euclidean_dim,
                    num_scale=opts.num_scale,
                )

                metrics = evaluate_samples(
                    samples, character_tokenizer,
                    euclidean_dim=euclidean_dim,
                    delta=opts.eval_delta,
                    num_scale=opts.num_scale,
                )
                metrics['iteration'] = training_iter

                print(
                    f'\n[Eval @ iter {training_iter}] '
                    f'accuracy={metrics["accuracy"]:.4f}  '
                    f'correct={metrics["correct"]}/{metrics["valid"]} valid  '
                    f'invalid={metrics["invalid"]}  '
                    f'mean_abs_error={metrics["mean_abs_error"]:.4f}'
                )

                with open(metrics_path, 'a') as mf:
                    mf.write(json.dumps(metrics) + '\n')

                if opts.enable_wandb:
                    import wandb
                    wandb.log({
                        'eval/accuracy': metrics['accuracy'],
                        'eval/mean_abs_error': metrics['mean_abs_error'],
                        'eval/valid': metrics['valid'],
                        'eval/invalid': metrics['invalid'],
                        'step': training_iter,
                    })

                model.train()
                ema.train()

    save_ckpt(model, ema, opt, scheduler, os.path.join(opts.dir, 'final_checkpoint.pt'))

    if opts.enable_wandb:
        import wandb
        wandb.finish()


if __name__ == '__main__':
    training()
