import json
import os
import click
import numpy as np
import torch
import wandb
from collections import OrderedDict
from copy import deepcopy
from torch.utils.data import DataLoader
from tqdm import tqdm
from multimodal_interpolant import MultimodalInterpolant
from autoregressive_interpolant import MultimodalInterpolant as AutoregressiveInterpolant
from branching_flows_interpolant import BranchingFlowsInterpolant
from custom_datasets.multimodal_math import ParenthesizedEquationsDataset
from utils.misc import dotdict
from utils.tokenizer import VocabTokenizer
from utils.optimizers import WarmUpScheduler
from models.mmdit_qm9 import MMDiTQM9
from visualize_dataset import plot_sample
from evaluate_equations_parenthesis import parse_equation
from multimodal_interpolant_both_var import MultimodalInterpolantBoth


def init_wandb(opts):
    wandb.init(
        project='MMDiT-Toy',
        name=f'toy-{opts.run_name}',
        tags=['training'],
        config=opts,
    )

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def evaluate_samples(samples, character_tokenizer, euclidean_dim=1, delta=1.0):
    """Compute accuracy metrics on a SamplingResult batch."""
    n = samples.xt.shape[0]
    valid = 0
    correct = 0
    invalid = 0
    diffs = []

    for i in range(n):
        symbols_str = character_tokenizer.decode(samples.yt[i].cpu())
        symbols = list(symbols_str)
        positions = samples.xt[i].cpu()[1:len(symbols) + 1, :]
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


def save_samples_jsonl(samples, character_tokenizer, path, euclidean_dim=1):
    """Write generated samples to a JSONL file compatible with evaluate_equations_parenthesis.py."""
    with open(path, 'w') as f:
        for i in range(samples.xt.shape[0]):
            symbols_str = character_tokenizer.decode(samples.yt[i].cpu())
            symbols = list(symbols_str)
            positions = samples.xt[i].cpu()[1:len(symbols) + 1, :]
            numbers = positions.squeeze(-1).tolist() if euclidean_dim == 1 else positions.tolist()
            f.write(json.dumps({'symbols': symbols, 'numbers': numbers, 'length': len(symbols)}) + '\n')


@click.command()
@click.option('--interpolant', type=click.Choice(['multimodal', 'autoregressive', 'branching', 'multimodal_both']), default='multimodal')
@click.option('--data_path', type=str, default=None)
@click.option('--optimizer', type=click.Choice(['adam', 'adamw']), default='adam')
@click.option('--ema_beta', type=float, default=.9999)
@click.option('--lr', type=float, default=1e-4)
@click.option('--batch_size', type=int, default=128)
@click.option('--log_rate', type=int, default=500)
@click.option('--num_iters', type=int, default=5000)
@click.option('--warmup_iters', type=int, default=100)
@click.option('--num_workers', type=int, default=2)
@click.option('--seed', type=int, default=42)
@click.option('--dir', type=str)
@click.option('--load_checkpoint', type=str, help='Path to a snapshot.pt to resume from')
@click.option('--enable_wandb', is_flag=True, default=False)
@click.option('--run_name', type=str, default='')
@click.option('--eval_samples', type=int, default=200, help='Number of samples to generate for evaluation')
@click.option('--eval_steps', type=int, default=20, help='Number of diffusion steps for evaluation sampling')
@click.option('--eval_delta', type=float, default=0.5, help='Absolute-error tolerance for counting a sample correct')
def training(**opts):
    opts = dotdict(opts)
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed(opts.seed)
    torch.cuda.manual_seed_all(opts.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    euclidean_dim = 1
    hidden_dim = 256
    character_tokenizer = VocabTokenizer(vocab={'+', '-', '*', '=', '.', '(', ')'})
    dataset = ParenthesizedEquationsDataset(character_tokenizer, data_path=opts.data_path)

    print('Vocab')
    print('--------------------------------')
    for token, idx in character_tokenizer.atom_to_idx.items():
        print(f'{token}: {idx}')
    print('--------------------------------')

    dataloader = DataLoader(
        dataset, batch_size=opts.batch_size, shuffle=True,
        num_workers=opts.num_workers, drop_last=True,
    )

    if opts.enable_wandb:
        init_wandb(opts)

    model = MMDiTQM9(
        branching_flows=opts.interpolant == 'branching',
        autoregressive=opts.interpolant == 'autoregressive',
        euclidean_dim=euclidean_dim,
        vocab_size=character_tokenizer.vocab_size,
        symbols_depth=4,
        positions_depth=4,
        depth=4,
        dim_modalities=[hidden_dim, hidden_dim],
        dim_joint_attn=hidden_dim,
        dim_conds=[hidden_dim, hidden_dim],
    ).to(device)
    ema = deepcopy(model)

    opt = torch.optim.AdamW(model.parameters(), lr=opts.lr)
    scheduler = WarmUpScheduler(opt, opts.warmup_iters)
    scaler = torch.amp.GradScaler()

    if opts.interpolant == 'multimodal':
        interpolant = MultimodalInterpolant(
            max_length=dataset.max_length,
            vocab_size=character_tokenizer.vocab_size,
            mask_token=character_tokenizer.mask_token_id,
            pad_token=character_tokenizer.pad_token_id,
            bos_token=character_tokenizer.bos_token_id,
            euclidean_dim=euclidean_dim,
        )
    elif opts.interpolant == 'autoregressive':
        interpolant = AutoregressiveInterpolant(
            max_length=dataset.max_length,
            vocab_size=character_tokenizer.vocab_size,
            mask_token=character_tokenizer.mask_token_id,
            pad_token=character_tokenizer.pad_token_id,
            bos_token=character_tokenizer.bos_token_id,
            euclidean_dim=euclidean_dim,
        )
    elif opts.interpolant == 'multimodal_both':
        interpolant = MultimodalInterpolantBoth(
            max_length=dataset.max_length,
            vocab_size=character_tokenizer.vocab_size,
            mask_token=character_tokenizer.mask_token_id,
            pad_token=character_tokenizer.pad_token_id,
            bos_token=character_tokenizer.bos_token_id,
            euclidean_dim=euclidean_dim,
        )
    elif opts.interpolant == 'branching':
        interpolant = BranchingFlowsInterpolant(
            vocab_size=character_tokenizer.vocab_size,
            mask_token=character_tokenizer.mask_token_id,
            pad_token=character_tokenizer.pad_token_id,
            euclidean_dim=euclidean_dim,
        )

    start_iter = 0
    if opts.load_checkpoint is not None:
        start_iter = load_checkpoint(opts, device, model, ema, opt, scheduler)

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

            opt.zero_grad()
            losses = interpolant.compute_loss(model, data_)

            if opts.interpolant in ('multimodal', 'autoregressive'):
                loss = (losses["dsm_loss"] + losses["discrete_unmasking_loss"]
                        + losses["euclidean_unmasking_loss"] + losses["insertion_loss"])
            elif opts.interpolant == 'multimodal_both':
                loss = (losses["dsm_loss"] + losses["discrete_unmasking_loss"]
                        + losses["euc_insertion_loss"] + losses["disc_insertion_loss"])
            elif opts.interpolant == 'branching':
                loss = (losses["dsm_loss"] + losses["discrete_unmasking_loss"]
                        + losses["insertion_loss"])

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
            loss_val = loss.detach().item()

            if opts.interpolant in ('multimodal', 'autoregressive'):
                pbar.set_description(
                    f'Iter {training_iter} --- '
                    f'DSM: {losses["dsm_loss"]:.4f}  '
                    f'Disc: {losses["discrete_unmasking_loss"]:.4f}  '
                    f'EucUnmask: {losses["euclidean_unmasking_loss"]:.4f}  '
                    f'Ins: {losses["insertion_loss"]:.4f}'
                )
            elif opts.interpolant == 'multimodal_both':
                pbar.set_description(
                    f'Iter {training_iter} --- '
                    f'DSM: {losses["dsm_loss"]:.4f}  '
                    f'Disc: {losses["discrete_unmasking_loss"]:.4f}  '
                    f'EucIns: {losses["euc_insertion_loss"]:.4f}  '
                    f'DiscIns: {losses["disc_insertion_loss"]:.4f}'
                )
            elif opts.interpolant == 'branching':
                pbar.set_description(
                    f'Iter {training_iter} --- '
                    f'DSM: {losses["dsm_loss"]:.4f}  '
                    f'Disc: {losses["discrete_unmasking_loss"]:.4f}  '
                    f'Ins: {losses["insertion_loss"]:.4f}'
                )

            if opts.enable_wandb:
                log_dict = {'loss': loss_val, 'step': training_iter}
                log_dict.update({k: v.item() for k, v in losses.items()})
                wandb.log(log_dict)

            if training_iter % opts.log_rate == 0 or training_iter == opts.num_iters:
                ckpt_path = os.path.join(opts.dir, f'itr_{training_iter}')
                os.makedirs(ckpt_path, exist_ok=True)
                save_ckpt(model, ema, opt, scheduler, os.path.join(ckpt_path, 'snapshot.pt'))

                model.eval()
                with torch.no_grad():
                    samples = interpolant.sampling(
                        model, opts.eval_steps, opts.eval_samples,
                        dataset.max_length, device, return_trace=False,
                    )

                # Save raw samples for offline analysis
                save_samples_jsonl(
                    samples, character_tokenizer,
                    os.path.join(ckpt_path, 'samples.jsonl'),
                    euclidean_dim=euclidean_dim,
                )

                # Evaluate accuracy
                metrics = evaluate_samples(
                    samples, character_tokenizer,
                    euclidean_dim=euclidean_dim,
                    delta=opts.eval_delta,
                )
                metrics['iteration'] = training_iter

                print(
                    f'\n[Eval @ iter {training_iter}] '
                    f'accuracy={metrics["accuracy"]:.4f}  '
                    f'correct={metrics["correct"]}/{metrics["valid"]} valid  '
                    f'invalid={metrics["invalid"]}  '
                    f'mean_abs_error={metrics["mean_abs_error"]:.4f}'
                )

                # Append to cumulative metrics log
                with open(metrics_path, 'a') as mf:
                    mf.write(json.dumps(metrics) + '\n')

                if opts.enable_wandb:
                    wandb.log({
                        'eval/accuracy': metrics['accuracy'],
                        'eval/mean_abs_error': metrics['mean_abs_error'],
                        'eval/valid': metrics['valid'],
                        'eval/invalid': metrics['invalid'],
                        'step': training_iter,
                    })

                model.train()

    save_ckpt(model, ema, opt, scheduler, os.path.join(opts.dir, 'final_checkpoint.pt'))

    if opts.enable_wandb:
        wandb.finish()


def load_checkpoint(opts, device, model, ema, opt, scheduler):
    snapshot = torch.load(opts.load_checkpoint, weights_only=True, map_location=device)
    model.load_state_dict(snapshot['model'], strict=False)
    ema.load_state_dict(snapshot['ema'], strict=False)
    opt.load_state_dict(snapshot['optimizer'])
    scheduler.load_state_dict(snapshot['scheduler'])
    return scheduler.last_epoch


def save_ckpt(model, ema, opt, scheduler, path):
    torch.save({
        'model': model.state_dict(),
        'ema': ema.state_dict(),
        'optimizer': opt.state_dict(),
        'scheduler': scheduler.state_dict(),
    }, path)


if __name__ == '__main__':
    training()
