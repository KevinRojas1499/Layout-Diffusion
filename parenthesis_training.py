import json
import os
import types
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


def _fill_equation_coefficients(sym_ids, coeffs_row, idx_to_atom, pad_id, bos_id):
    """Fill per-number-slot coefficients for the L-R=0 constraint (in-place).

    The data format: first term's sign is embedded in the number value; subsequent
    terms' signs come from the binary +/- operators in the symbol sequence (parens
    are structural only and do not affect sign assignment).
    """
    syms = []
    for tid in sym_ids:
        tid = int(tid)
        if tid in (pad_id, bos_id):
            continue
        tok = idx_to_atom.get(tid)
        if tok is None:
            continue
        syms.append(tok)

    eq_idx = None
    dot_idx = None
    for i, tok in enumerate(syms):
        if tok == '=' and eq_idx is None:
            eq_idx = i
        if tok == '.':
            dot_idx = i
            break

    if eq_idx is None:
        return

    end_idx = dot_idx if dot_idx is not None else len(syms)
    left_ops = [t for t in syms[:eq_idx] if t in ('+', '-')]
    right_ops = [t for t in syms[eq_idx + 1:end_idx] if t in ('+', '-')]

    L = len(coeffs_row)
    # Left side: position 1 = first number (sign embedded → coeff +1),
    #            positions 2..n_left = subsequent numbers (coeff from operator)
    if 1 < L:
        coeffs_row[1] = 1.0
    for i, op in enumerate(left_ops):
        pos = 2 + i
        if pos < L:
            coeffs_row[pos] = 1.0 if op == '+' else -1.0

    # Right side: negated for L - R = 0
    n_left = 1 + len(left_ops)
    rs = 1 + n_left  # right-side start position
    if rs < L:
        coeffs_row[rs] = -1.0
    for i, op in enumerate(right_ops):
        pos = rs + 1 + i
        if pos < L:
            coeffs_row[pos] = -1.0 if op == '+' else 1.0


def compute_constraint_coefficients_batch(y1, idx_to_atom, pad_id, bos_id):
    """Return [batch, seq_len] float tensor of per-number-slot coefficients."""
    B, L = y1.shape
    coeffs = torch.zeros(B, L, dtype=torch.float32, device=y1.device)
    y1_cpu = y1.cpu().tolist()
    for b in range(B):
        _fill_equation_coefficients(y1_cpu[b], coeffs[b], idx_to_atom, pad_id, bos_id)
    return coeffs


def gaussian_mmd(x, y, bandwidths=(0.5, 1.0, 2.0, 5.0)):
    """Multi-scale unbiased MMD² with Gaussian kernels.

    Compares marginal distributions of predicted vs training active numbers.
    Penalizes scale shrinkage: training numbers have std~2.5 (l5), so predicting
    near-zero numbers gives high MMD, pushing values toward training scale.
    x, y: 1D float tensors of sampled values from pred and target distributions.
    """
    if x.numel() < 2 or y.numel() < 2:
        return x.sum() * 0.0
    mmd_total = x.new_zeros(1)
    n, m = x.shape[0], y.shape[0]
    for bw in bandwidths:
        inv_bw2 = 1.0 / (2.0 * bw * bw)
        kxx = torch.exp(-(x.unsqueeze(1) - x.unsqueeze(0)).pow(2) * inv_bw2)
        kyy = torch.exp(-(y.unsqueeze(1) - y.unsqueeze(0)).pow(2) * inv_bw2)
        kxy = torch.exp(-(x.unsqueeze(1) - y.unsqueeze(0)).pow(2) * inv_bw2)
        mmd_bw = ((kxx.sum() - kxx.diag().sum()) / (n * (n - 1))
                  + (kyy.sum() - kyy.diag().sum()) / (m * (m - 1))
                  - 2.0 * kxy.mean())
        mmd_total = mmd_total + mmd_bw
    return (mmd_total / len(bandwidths)).squeeze()


class ConstraintHead(torch.nn.Module):
    """Per-position balance-correction MLP.

    Takes (x_i, coeff_i, global_residual) per position and predicts delta_i.
    Trained by the constraint loss on (x_backbone.detach() + delta), so it
    learns to distribute the residual without affecting backbone scale.
    At inference: corrected_x = backbone_x + head(backbone_x, coeffs).
    """
    def __init__(self, hidden=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, x0, coeffs):
        """x0: [B, L, 1], coeffs: [B, L] → delta: [B, L, 1]"""
        residual = (coeffs * x0.squeeze(-1)).sum(dim=1, keepdim=True)  # [B, 1]
        res_bc = residual.unsqueeze(1).expand(-1, x0.size(1), 1)       # [B, L, 1]
        inp = torch.cat([x0, coeffs.unsqueeze(-1), res_bc], dim=-1)    # [B, L, 3]
        return self.net(inp)                                             # [B, L, 1]


class HeadWrappedModel(torch.nn.Module):
    """Wraps backbone + ConstraintHead for per-step inference correction.

    At each denoising step, the backbone's x0 prediction is corrected by the head
    before being used to compute xt-1. This matches the training distribution (where
    the head corrects x0_pred at each training timestep) to the inference distribution.
    """
    def __init__(self, backbone, head, idx_to_atom, pad_id, bos_id):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.idx_to_atom = idx_to_atom
        self.pad_id = pad_id
        self.bos_id = bos_id

    def forward(self, **kwargs):
        prediction = self.backbone(**kwargs)
        yt = kwargs['cat_tokens']  # [B, L] — current discrete token state
        coeffs = compute_constraint_coefficients_batch(yt, self.idx_to_atom, self.pad_id, self.bos_id)
        delta = self.head(prediction.clean_data, coeffs)
        prediction.clean_data = prediction.clean_data + delta
        return prediction


def make_constraint_loss_fn(idx_to_atom, pad_id, bos_id, scale_weight=0.0, use_hinge=False,
                            use_normalized=False, threshold_getter=None, normalized_scale_target=1.0,
                            use_proxy=False, scale_hinge_target=0.0, constraint_head=None,
                            nonzero_floor=0.0, nonzero_weight=0.0, use_relative=False, relative_floor=0.5,
                            mmd_weight=0.0, mmd_bandwidths=(0.5, 1.0, 2.0, 5.0)):
    """Return a callback suitable for MultimodalInterpolant.compute_loss(extra_loss_fn=...).

    use_hinge: replace squared residual with max(0, |L-R|-threshold)^2. threshold is fixed at 0.5
        unless threshold_getter is provided (see below).
    threshold_getter: callable () -> float that returns the current hinge threshold. Used for
        curriculum training: start with a loose threshold (e.g. 2.0) and tighten to 0.5 over time.
        Implies use_hinge=True.
    normalized_scale_target: minimum value for the active_scale clamp in use_normalized mode.
    scale_weight: weight for scale_hinge_loss when scale_hinge_target > 0.
    scale_hinge_target: if > 0, add scale_weight * max(0, S - mean|x_active|)^2 to oppose scale
        shrinkage. Unlike the old reward (-mean|x|), the hinge is bounded: gradient is zero when
        mean|x_active| >= S, preventing runaway. Equilibrium is at mean|x_active| = S.
    use_normalized: divide residual by sum of |x| at active (non-pad) positions — makes loss
        scale-invariant so the trivial x=0 solution has no gradient advantage.
    use_relative: divide residual by per-sample mean|x_active| (detached, floored at relative_floor).
        Unlike use_normalized (which divides by sum over all positions), this uses mean over active
        positions only. Key property: the gradient is WEAKER at large scale (sigma in denominator),
        so the model can maintain l5-scale numbers with less constraint disruption. At small scale
        (sigma → floor), the gradient is stronger than standard, opposing collapse to x=0.
    relative_floor: minimum denominator for use_relative mode (default 0.5). Prevents division by
        near-zero when scale collapses; makes gradient at x=0 equal to standard constraint * (1/floor)^2.
    nonzero_floor: if > 0, add per-position hinge mean_active(max(0, floor - |x_i|)^2) to penalize
        active positions with small magnitude. Directly blocks the "predict zeros" degenerate solution.
    nonzero_weight: weight for the per-position nonzero hinge. Typical value: 0.5–2.0.
    """
    def constraint_fn(prediction, interpolant_sample, x1, y1):
        coeffs = compute_constraint_coefficients_batch(y1, idx_to_atom, pad_id, bos_id)
        # Un-reorder model's x0 prediction back to original token order
        inv_st = interpolant_sample.st.argsort(dim=1)
        x0_orig = prediction.clean_data.gather(
            1, inv_st.unsqueeze(-1).expand_as(prediction.clean_data)
        )  # [B, L, D]
        if constraint_head is not None:
            # Head mode: backbone is detached — only the head receives constraint gradient.
            # The head learns to redistribute residual without affecting backbone scale.
            x0_detached = x0_orig.detach()
            delta = constraint_head(x0_detached, coeffs)
            x0_corrected = x0_detached + delta
            residual_head = (coeffs * x0_corrected.squeeze(-1)).sum(dim=1)
            return {"constraint_loss": residual_head.pow(2).mean()}
        residual = (coeffs * x0_orig.squeeze(-1)).sum(dim=1)  # [B]
        active_mask = (coeffs.abs() > 0.5).float()  # [B, L] — number positions only
        num_active = active_mask.sum(dim=1).clamp(min=1.0)  # [B]
        if use_relative:
            # Relative constraint: (L-R)^2 / sigma^2 where sigma = mean|x_active|, floored.
            # Weaker gradient at large scale → DSM has more room to maintain l5-scale numbers.
            # Stronger gradient at small scale (sigma=floor) → opposes collapse to x=0.
            sigma = (x0_orig.squeeze(-1).abs() * active_mask).sum(dim=1) / num_active  # [B]
            sigma = sigma.detach().clamp(min=relative_floor)
            balance_loss = (residual / sigma).pow(2).mean()
        elif use_normalized:
            active_scale = (x0_orig.squeeze(-1).abs() * active_mask).sum(dim=1).detach()  # [B]
            scale = active_scale.clamp(min=normalized_scale_target)
            balance_loss = (residual / scale).pow(2).mean()
        elif use_proxy:
            # Proxy constraint: detach residual to remove x_i self-coupling.
            # gradient = residual_detached * coeff_i (no x_i² term → no pull toward zero).
            balance_loss = (residual.detach() * residual).mean()
        elif threshold_getter is not None or use_hinge:
            threshold = threshold_getter() if threshold_getter is not None else 0.5
            balance_loss = torch.clamp(residual.abs() - threshold, min=0.0).pow(2).mean()
        else:
            balance_loss = residual.pow(2).mean()
        if scale_hinge_target > 0.0 and scale_weight > 0.0:
            # Scale hinge: penalize mean|x_active| below target. Zero gradient above target → no runaway.
            # Equilibrium: constraint balances equations while scale hinge keeps magnitudes near target.
            active_mean_abs = (x0_orig.squeeze(-1).abs() * active_mask).sum(dim=1) / num_active  # [B]
            scale_hinge = torch.clamp(scale_hinge_target - active_mean_abs, min=0.0).pow(2).mean()
            balance_loss = balance_loss + scale_weight * scale_hinge
        elif scale_weight > 0.0 and not use_normalized:
            # Legacy: direct reward (unbounded — prefer scale_hinge_target instead)
            balance_loss = balance_loss - scale_weight * x0_orig.squeeze(-1).abs().mean()
        if nonzero_floor > 0.0 and nonzero_weight > 0.0:
            # Per-position nonzero hinge: penalizes each active position with |x| < floor.
            # Blocks "predict two matching anchors + zeros" degenerate solution.
            per_pos_hinge = torch.clamp(nonzero_floor - x0_orig.squeeze(-1).abs(), min=0.0).pow(2)
            nonzero_loss = (per_pos_hinge * active_mask).sum(dim=1) / num_active
            balance_loss = balance_loss + nonzero_weight * nonzero_loss.mean()
        if mmd_weight > 0.0:
            # MMD on active number marginals: predicted vs training data values.
            # Penalizes scale shrinkage — trivial x=0 solution has high MMD since
            # training numbers have std~2.5, not 0. Forces predicted distribution
            # toward training data scale without a trivial minimum at x=0.
            x0_active = x0_orig.squeeze(-1)[active_mask.bool()]
            x1_active = x1.squeeze(-1)[active_mask.bool()]
            mmd = gaussian_mmd(x0_active, x1_active, bandwidths=mmd_bandwidths)
            balance_loss = balance_loss + mmd_weight * mmd
        return {"constraint_loss": balance_loss}
    return constraint_fn


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


def evaluate_samples(samples, character_tokenizer, euclidean_dim=1, delta=0.5, num_scale=1.0):
    """Compute accuracy metrics on a SamplingResult batch.

    num_scale: multiply generated positions by this before evaluating (reverses training normalization).
    """
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
    """Write generated samples to a JSONL file compatible with evaluate_equations_parenthesis.py."""
    with open(path, 'w') as f:
        for i in range(samples.xt.shape[0]):
            symbols_str = character_tokenizer.decode(samples.yt[i].cpu())
            symbols = list(symbols_str)
            positions = samples.xt[i].cpu()[1:len(symbols) + 1, :] * num_scale
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
@click.option('--load_checkpoint', type=str, help='Path to a snapshot.pt to resume from (restores model+optimizer+scheduler)')
@click.option('--finetune_checkpoint', type=str, default=None,
              help='Load model weights only from this snapshot (fresh optimizer/scheduler — for curriculum learning)')
@click.option('--enable_wandb', is_flag=True, default=False)
@click.option('--run_name', type=str, default='')
@click.option('--eval_samples', type=int, default=200, help='Number of samples to generate for evaluation')
@click.option('--eval_steps', type=int, default=20, help='Number of diffusion steps for evaluation sampling')
@click.option('--eval_delta', type=float, default=0.5, help='Absolute-error tolerance for counting a sample correct')
@click.option('--lr_schedule', type=click.Choice(['warmup_only', 'cosine']), default='warmup_only',
              help='LR schedule: warmup_only keeps LR constant after warmup; cosine decays to lr_min after warmup')
@click.option('--lr_min', type=float, default=1e-6, help='Minimum LR for cosine schedule')
@click.option('--num_scale', type=float, default=1.0,
              help='Divide training numbers by this factor (e.g. 5.0 for l5 data) and multiply back at eval')
@click.option('--constraint_weight', type=float, default=0.0,
              help='Weight for differentiable equation-balance constraint loss (L-R)^2. 0 disables it.')
@click.option('--scale_weight', type=float, default=0.0,
              help='Weight for scale-reward term: subtracts scale_weight * mean(|x_i|) from constraint loss to oppose scale shrinkage.')
@click.option('--use_hinge', is_flag=True, default=False,
              help='Use hinge constraint max(0,|L-R|-0.5)^2 instead of squared residual.')
@click.option('--use_normalized', is_flag=True, default=False,
              help='Divide residual by sum(|x| at active positions) — scale-invariant constraint that avoids x=0 bias.')
@click.option('--normalized_scale_target', type=float, default=1.0,
              help='Soft-cap floor for the active_scale denominator in --use_normalized mode. Set to ~15 for l5 anchoring. Use constraint_weight = base_w * target^2 (e.g. target=15 -> weight=450 for base 2.0).')
@click.option('--use_proxy', is_flag=True, default=False,
              help='Proxy constraint: detach residual in one factor to remove x_i self-coupling (scale shrinkage). Gradient = residual_sg*coeff_i instead of 2*residual*coeff_i. Use constraint_weight*=2 to compensate for halved gradient.')
@click.option('--scale_hinge_target', type=float, default=0.0,
              help='Scale hinge target S: adds scale_weight * max(0, S - mean|x_active|)^2 to constraint loss. Zero gradient above S prevents runaway. Use scale_weight to control the hinge strength.')
@click.option('--hinge_threshold_start', type=float, default=0.5,
              help='Starting hinge threshold for curriculum training (e.g. 2.0). Decays linearly to 0.5 over hinge_curriculum_iters.')
@click.option('--hinge_curriculum_iters', type=int, default=0,
              help='Iters over which hinge threshold decays from hinge_threshold_start to 0.5. 0 = no curriculum (fixed threshold).')
@click.option('--nonzero_floor', type=float, default=0.0,
              help='Per-position nonzero hinge floor: penalize active positions with |x| < floor via mean_active(max(0,floor-|x|)^2). Blocks predict-zeros degenerate solution.')
@click.option('--nonzero_weight', type=float, default=0.0,
              help='Weight for the per-position nonzero hinge loss. Typical range: 0.5–2.0.')
@click.option('--use_relative', is_flag=True, default=False,
              help='Relative constraint: divide residual by per-sample mean|x_active| (floored at relative_floor). Weaker gradient at large scale; stronger at small scale. Distinct from --use_normalized (which divides by sum, not mean).')
@click.option('--relative_floor', type=float, default=0.5,
              help='Floor for mean|x_active| denominator in --use_relative mode. Prevents division by near-zero at small scale.')
@click.option('--mmd_weight', type=float, default=0.0,
              help='Weight for MMD loss on active number marginals (predicted vs training). Penalizes scale shrinkage.')
@click.option('--mmd_bandwidths', type=str, default='0.5,1.0,2.0,5.0',
              help='Comma-separated Gaussian kernel bandwidths for MMD loss.')
@click.option('--depth', type=int, default=4, help='Number of joint attention blocks in MMDiTQM9')
@click.option('--hidden_dim', type=int, default=256, help='Embedding dim for symbols and positions')
@click.option('--use_constraint_head', is_flag=True, default=False,
              help='Use separate ConstraintHead MLP trained only by constraint loss; backbone trains on DSM only, preventing scale shrinkage.')
def training(**opts):
    opts = dotdict(opts)
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed(opts.seed)
    torch.cuda.manual_seed_all(opts.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    euclidean_dim = 1
    hidden_dim = opts.hidden_dim
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
        symbols_depth=opts.depth,
        positions_depth=opts.depth,
        depth=opts.depth,
        dim_modalities=[hidden_dim, hidden_dim],
        dim_joint_attn=hidden_dim,
        dim_conds=[hidden_dim, hidden_dim],
    ).to(device)
    ema = deepcopy(model)

    constraint_head = None
    ema_head = None
    if opts.constraint_weight > 0 and opts.use_constraint_head:
        constraint_head = ConstraintHead(hidden=64).to(device)
        ema_head = deepcopy(constraint_head)
        print('ConstraintHead enabled: backbone trains on DSM only; head trains on constraint only.')

    if constraint_head is not None:
        opt = torch.optim.AdamW([
            {'params': model.parameters(), 'lr': opts.lr},
            {'params': constraint_head.parameters(), 'lr': 1e-3},
        ])
    else:
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
        start_iter = load_checkpoint(opts, device, model, ema, opt, scheduler,
                                     constraint_head=constraint_head, ema_head=ema_head)
    elif opts.finetune_checkpoint is not None:
        snap = torch.load(opts.finetune_checkpoint, weights_only=True, map_location=device)
        model.load_state_dict(snap['model'], strict=False)
        ema.load_state_dict(snap['ema'], strict=False)
        print(f'Loaded weights from {opts.finetune_checkpoint} (fresh optimizer)')

    constraint_fn = None
    threshold_getter = None
    if opts.constraint_weight > 0 and opts.interpolant in ('multimodal', 'autoregressive'):
        threshold_getter = None
        if opts.hinge_threshold_start > 0.5 and opts.hinge_curriculum_iters > 0:
            # Mutable state for curriculum threshold; updated in the training loop below
            _threshold = [float(opts.hinge_threshold_start)]
            threshold_getter = lambda: _threshold[0]
            print(f'Hinge curriculum: threshold {opts.hinge_threshold_start} → 0.5 over {opts.hinge_curriculum_iters} iters')
        constraint_fn = make_constraint_loss_fn(
            character_tokenizer.idx_to_atom,
            character_tokenizer.pad_token_id,
            character_tokenizer.bos_token_id,
            scale_weight=opts.scale_weight,
            use_hinge=opts.use_hinge,
            use_normalized=opts.use_normalized,
            threshold_getter=threshold_getter,
            normalized_scale_target=opts.normalized_scale_target,
            use_proxy=opts.use_proxy,
            scale_hinge_target=opts.scale_hinge_target,
            constraint_head=constraint_head,
            nonzero_floor=opts.nonzero_floor,
            nonzero_weight=opts.nonzero_weight,
            use_relative=opts.use_relative,
            relative_floor=opts.relative_floor,
            mmd_weight=opts.mmd_weight,
            mmd_bandwidths=tuple(float(b) for b in opts.mmd_bandwidths.split(',')),
        )
        print(f'Constraint loss enabled: weight={opts.constraint_weight}, use_hinge={opts.use_hinge}, use_normalized={opts.use_normalized}, use_proxy={opts.use_proxy}, scale_hinge_target={opts.scale_hinge_target}, use_constraint_head={opts.use_constraint_head}, nonzero_floor={opts.nonzero_floor}, nonzero_weight={opts.nonzero_weight}, use_relative={opts.use_relative}, relative_floor={opts.relative_floor}, mmd_weight={opts.mmd_weight}, mmd_bandwidths={opts.mmd_bandwidths}')

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

            # Update curriculum hinge threshold if applicable
            if threshold_getter is not None:
                progress = min(1.0, training_iter / opts.hinge_curriculum_iters)
                _threshold[0] = opts.hinge_threshold_start - progress * (opts.hinge_threshold_start - 0.5)

            opt.zero_grad()
            losses = interpolant.compute_loss(model, data_, extra_loss_fn=constraint_fn)

            if opts.interpolant in ('multimodal', 'autoregressive'):
                loss = (losses["dsm_loss"] + losses["discrete_unmasking_loss"]
                        + losses["euclidean_unmasking_loss"] + losses["insertion_loss"])
                if constraint_fn is not None:
                    loss = loss + opts.constraint_weight * losses["constraint_loss"]
            elif opts.interpolant == 'multimodal_both':
                loss = (losses["dsm_loss"] + losses["discrete_unmasking_loss"]
                        + losses["euc_insertion_loss"] + losses["disc_insertion_loss"])
            elif opts.interpolant == 'branching':
                loss = (losses["dsm_loss"] + losses["discrete_unmasking_loss"]
                        + losses["insertion_loss"])

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            update_ema(ema, model, decay=opts.ema_beta)
            if ema_head is not None:
                update_ema(ema_head, constraint_head, decay=opts.ema_beta)
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
                desc = (
                    f'Iter {training_iter} --- '
                    f'DSM: {losses["dsm_loss"]:.4f}  '
                    f'Disc: {losses["discrete_unmasking_loss"]:.4f}  '
                    f'EucUnmask: {losses["euclidean_unmasking_loss"]:.4f}  '
                    f'Ins: {losses["insertion_loss"]:.4f}'
                )
                if constraint_fn is not None:
                    desc += f'  Cstr: {losses["constraint_loss"]:.4f}'
                pbar.set_description(desc)
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
                save_ckpt(model, ema, opt, scheduler, os.path.join(ckpt_path, 'snapshot.pt'),
                          constraint_head=constraint_head, ema_head=ema_head)

                # Use EMA for evaluation — EMA tracks the smoothed model and gives
                # better sample quality, but only once ema_beta^n_iters ≈ 0.
                ema.eval()
                if ema_head is not None:
                    ema_head.eval()
                    # Per-step head correction: apply head at each denoising step.
                    # This matches the training distribution (head sees x0_pred at each timestep).
                    eval_model = HeadWrappedModel(
                        ema, ema_head,
                        character_tokenizer.idx_to_atom,
                        character_tokenizer.pad_token_id,
                        character_tokenizer.bos_token_id,
                    )
                else:
                    eval_model = ema
                with torch.no_grad():
                    samples = interpolant.sampling(
                        eval_model, opts.eval_steps, opts.eval_samples,
                        dataset.max_length, device, return_trace=False,
                    )
                eval_samples = samples

                # Save raw samples for offline analysis
                save_samples_jsonl(
                    eval_samples, character_tokenizer,
                    os.path.join(ckpt_path, 'samples.jsonl'),
                    euclidean_dim=euclidean_dim,
                    num_scale=opts.num_scale,
                )

                # Evaluate accuracy
                metrics = evaluate_samples(
                    eval_samples, character_tokenizer,
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
                ema.train()
                if ema_head is not None:
                    constraint_head.train()
                    ema_head.train()
                    # eval_model was a HeadWrappedModel wrapper (not a persistent module) — no cleanup needed

    save_ckpt(model, ema, opt, scheduler, os.path.join(opts.dir, 'final_checkpoint.pt'),
              constraint_head=constraint_head, ema_head=ema_head)

    if opts.enable_wandb:
        wandb.finish()


def load_checkpoint(opts, device, model, ema, opt, scheduler, constraint_head=None, ema_head=None):
    snapshot = torch.load(opts.load_checkpoint, weights_only=True, map_location=device)
    model.load_state_dict(snapshot['model'], strict=False)
    ema.load_state_dict(snapshot['ema'], strict=False)
    opt.load_state_dict(snapshot['optimizer'])
    scheduler.load_state_dict(snapshot['scheduler'])
    if constraint_head is not None and 'constraint_head' in snapshot:
        constraint_head.load_state_dict(snapshot['constraint_head'])
    if ema_head is not None and 'ema_head' in snapshot:
        ema_head.load_state_dict(snapshot['ema_head'])
    # SequentialLR stores last_epoch on each sub-scheduler; WarmUpScheduler stores it directly.
    last = snapshot['scheduler'].get('last_epoch', None)
    return last if last is not None else scheduler.last_epoch


def save_ckpt(model, ema, opt, scheduler, path, constraint_head=None, ema_head=None):
    d = {
        'model': model.state_dict(),
        'ema': ema.state_dict(),
        'optimizer': opt.state_dict(),
        'scheduler': scheduler.state_dict(),
    }
    if constraint_head is not None:
        d['constraint_head'] = constraint_head.state_dict()
    if ema_head is not None:
        d['ema_head'] = ema_head.state_dict()
    torch.save(d, path)


if __name__ == '__main__':
    training()
