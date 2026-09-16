"""Layout training with Hydra config."""

import json
import os
import torch
import wandb
from collections import OrderedDict
from copy import deepcopy
from torch.utils.data import DataLoader
from tqdm import tqdm

import hydra
from omegaconf import OmegaConf

from multimodal_interpolant import MultimodalInterpolant
from continuous_masking_interpolant import ContinuousMaskingInterpolant
from analog_bit import AnalogBit
from custom_datasets.layout_labels import DATASET_REGISTRY as LAYOUT_REGISTRY, build_tokenizer as build_layout_tokenizer
from custom_datasets.layoutflow_h5 import LayoutFlowH5Dataset
from eval.layout import LayoutEvaluator
from utils.tokenizer import VocabTokenizer
from utils.optimizers import WarmUpScheduler, CosineDecayScheduler, CombinedOptimizer
from models.transformer import Transformer
from visualize_dataset import plot_layout_sample

from conf.schema import register_configs, LayoutConfigSchema

# Register configs before Hydra composes (required for schema validation)
register_configs()


def init_wandb(cfg: LayoutConfigSchema) -> None:
    """Initialize Weights & Biases logging."""
    run = cfg.run
    wandb.init(
        project="MMDiT-Layout",
        name=f"layout-{run.run_name}" if run.run_name else "layout",
        tags=["training", "layout"],
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    wandb.define_metric("step")
    wandb.define_metric("*", step_metric="step")


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def _get_dataset_and_dims(cfg: LayoutConfigSchema):
    """Build dataset and return (dataset, euclidean_dim, hidden_dim)."""
    ds_cfg = cfg.dataset
    max_length = ds_cfg.max_length

    if ds_cfg.name not in LAYOUT_REGISTRY:
        raise ValueError(f"Unknown dataset: {ds_cfg.name}")
    tokenizer = build_layout_tokenizer(ds_cfg.name)
    dataset = LayoutFlowH5Dataset(
        tokenizer=tokenizer,
        dataset_name=ds_cfg.name,
        split=ds_cfg.split,
        max_length=max_length,
        data_path=ds_cfg.data_path,
    )
    euclidean_dim = 4  # bbox: x_center, y_center, w, h
    hidden_dim = cfg.model.hidden_dim

    return dataset, tokenizer, euclidean_dim, hidden_dim, max_length


def _get_model(cfg: LayoutConfigSchema, vocab_size: int, euclidean_dim: int, hidden_dim: int, device: torch.device):
    """Build model from config."""
    m_cfg = cfg.model
    if m_cfg.name != "Transformer":
        raise ValueError(f"Unknown model: {m_cfg.name}")
    model = Transformer(
        euclidean_dim=euclidean_dim,
        vocab_size=vocab_size,
        dim=hidden_dim,
        depth=m_cfg.depth,
        use_rope=m_cfg.use_rope,
    )
    return model.to(device)


def _get_optimizer(cfg: LayoutConfigSchema, model: torch.nn.Module) -> torch.optim.Optimizer:
    """Build optimizer from config."""
    opt_cfg = cfg.optimizer
    if opt_cfg.name == "adam":
        return torch.optim.Adam(model.parameters(), lr=opt_cfg.lr)
    elif opt_cfg.name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.weight_decay,
        )
    elif opt_cfg.name == "muon":
        param_groups = model.get_muon_adam_params()
        muon_params = param_groups["muon_params"]
        adam_params = param_groups["adam_params"]
        muon_lr = getattr(opt_cfg, "muon_lr", 5e-4)
        muon_weight_decay = getattr(opt_cfg, "muon_weight_decay", 0.01)
        muon_opt = torch.optim.Muon(
            muon_params, lr=muon_lr, weight_decay=muon_weight_decay, adjust_lr_fn="match_rms_adamw"
        )
        adam_opt = torch.optim.AdamW(adam_params, lr=opt_cfg.lr, weight_decay=opt_cfg.weight_decay)
        return CombinedOptimizer(muon_opt, adam_opt)
    raise ValueError(f"Unknown optimizer: {opt_cfg.name}")


def _get_interpolant(cfg: LayoutConfigSchema, dataset, tokenizer, euclidean_dim: int):
    """Build interpolant from config."""
    int_cfg = cfg.interpolant
    max_length = dataset.max_length

    if int_cfg.name == "continuous_masking":
        # euclidean_dim already includes the analog-bit category width -- see
        # main()'s widening of it before this is called for this interpolant.
        return ContinuousMaskingInterpolant(euclidean_dim=euclidean_dim)

    if int_cfg.name != "multimodal":
        raise ValueError(f"Unknown interpolant: {int_cfg.name}")
    return MultimodalInterpolant(
        max_length=max_length,
        vocab_size=tokenizer.vocab_size,
        mask_token=tokenizer.mask_token_id,
        pad_token=tokenizer.pad_token_id,
        bos_token=tokenizer.bos_token_id,
        euclidean_dim=euclidean_dim,
        dsm_t_reweight=int_cfg.dsm_t_reweight,
        cfg_dropout_prob=int_cfg.cfg_dropout_prob,
        cat_cond_prob=int_cfg.cat_cond_prob,
        size_cond_prob=int_cfg.size_cond_prob,
        fixed_length=int_cfg.fixed_length,
    )


def _build_evaluator(cfg: LayoutConfigSchema, tokenizer: VocabTokenizer, device: torch.device):
    """Construct a LayoutEvaluator if eval is enabled and dataset is supported."""
    eval_cfg = getattr(cfg, "eval", None)
    if eval_cfg is None or not getattr(eval_cfg, "enabled", False):
        return None
    if cfg.interpolant.name == "continuous_masking":
        # No FID wiring yet for this interpolant -- see periodic-visualization
        # branch in main() for what it does get (plain PNG samples).
        return None
    if cfg.dataset.name not in LAYOUT_REGISTRY:
        return None
    return LayoutEvaluator.for_dataset(
        dataset_name=cfg.dataset.name,
        tokenizer=tokenizer,
        device=device,
        layoutflow_root=getattr(eval_cfg, "layoutflow_root", None),
    )


def _build_fixed_length_mask(lengths: torch.Tensor, max_length: int, device: torch.device) -> torch.Tensor:
    """Build a [B, max_length+1] bool mask: BOS + first `lengths[i]` positions True."""
    B = lengths.shape[0]
    L = max_length + 1
    idx = torch.arange(L, device=device).unsqueeze(0)  # [1, L]
    lens = lengths.to(device).unsqueeze(1)             # [B, 1]
    # Position 0 is BOS (always active); positions 1..len_i are valid.
    return (idx == 0) | ((idx >= 1) & (idx <= lens))


@torch.no_grad()
def _run_eval(evaluator, interpolant, model, eval_cfg, max_length, device, train_lengths=None):
    """Generate `eval_cfg.num_samples` and compute FID + Alignment metrics."""
    if getattr(interpolant, "fixed_length", False):
        # Sample lengths from the training distribution, build masks, then run
        # the fixed-length sampler. No insertions/deletions involved.
        if train_lengths is None or train_lengths.numel() == 0:
            raise ValueError("fixed_length eval requires train_lengths to be passed in.")
        n = eval_cfg.num_samples
        idx = torch.randint(0, train_lengths.shape[0], (n,), device=train_lengths.device)
        lens = train_lengths[idx]
        mask = _build_fixed_length_mask(lens, max_length, device)
        samples = interpolant.fixed_length_sampling(
            model, mask, eval_cfg.num_steps, device,
            sampler=eval_cfg.sampler, return_trace=False,
        )
        return evaluator.evaluate_samples(
            samples.xt, samples.yt, samples.mask_t, batch_size=eval_cfg.batch_size,
        )
    samples = interpolant.sampling(
        model,
        eval_cfg.num_steps,
        eval_cfg.num_samples,
        max_length + 1,
        device,
        return_trace=False,
        sampler=eval_cfg.sampler,
    )
    mask = samples.y_mask_t if hasattr(samples, "y_mask_t") else samples.mask_t
    return evaluator.evaluate_samples(
        samples.xt, samples.yt, mask, batch_size=eval_cfg.batch_size,
    )


def load_checkpoint_state(
    load_path: str,
    device: torch.device,
    model: torch.nn.Module,
    ema: torch.nn.Module,
    opt: torch.optim.Optimizer,
) -> tuple[int, dict]:
    """Load model+ema+optimizer state and return (start_iter, scheduler_state).

    Scheduler state is returned separately so the caller can either rehydrate
    the original scheduler or build a new one (e.g. swap to cosine decay on
    resume).
    """
    snapshot = torch.load(load_path, weights_only=True)
    model.load_state_dict(snapshot["model"], strict=False)
    ema.load_state_dict(snapshot["ema"], strict=False)
    opt.load_state_dict(snapshot["optimizer"])
    sched_state = snapshot["scheduler"]
    start_iter = int(sched_state.get("last_epoch", -1)) + 1
    return start_iter, sched_state


def save_ckpt(
    model: torch.nn.Module,
    ema: torch.nn.Module,
    opt: torch.optim.Optimizer,
    scheduler: WarmUpScheduler,
    path: str,
) -> None:
    """Save checkpoint."""
    snapshot = {
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    torch.save(snapshot, path)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: LayoutConfigSchema) -> None:
    """Main training entry point."""
    run_cfg = cfg.run
    torch.manual_seed(run_cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(run_cfg.seed)
        torch.cuda.manual_seed_all(run_cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset, tokenizer, euclidean_dim, hidden_dim, max_length = _get_dataset_and_dims(cfg)

    # continuous_masking has no discrete category channel: y gets analog-bit
    # encoded and concatenated onto x, and the model's discrete cat_tokens
    # pathway (vocab_size=2) is repurposed to carry the masked-state flag
    # instead of a category id -- see continuous_masking_interpolant.py.
    is_continuous_category = cfg.interpolant.name == "continuous_masking"
    analog_bit = AnalogBit(num_cat=tokenizer.vocab_size) if is_continuous_category else None
    model_vocab_size = 2 if is_continuous_category else tokenizer.vocab_size
    if is_continuous_category:
        euclidean_dim = euclidean_dim + analog_bit.num_bits

    collate_fn = getattr(dataset, "dynamic_collate", None)
    dataloader = DataLoader(
        dataset,
        batch_size=run_cfg.batch_size,
        shuffle=True,
        num_workers=run_cfg.num_workers,
        drop_last=True,
        collate_fn=collate_fn,
    )

    if run_cfg.enable_wandb:
        init_wandb(cfg)

    model = _get_model(cfg, model_vocab_size, euclidean_dim, hidden_dim, device)
    ema = deepcopy(model)
    opt = _get_optimizer(cfg, model)
    interpolant = _get_interpolant(cfg, dataset, tokenizer, euclidean_dim)

    # Cache the training-set length tensor for fixed_length eval. LayoutFlowH5Dataset
    # exposes per-sample lengths via `_lens` (truncated at max_length).
    train_lengths = getattr(dataset, "_lens", None)
    if train_lengths is not None:
        train_lengths = train_lengths.clone()

    evaluator = _build_evaluator(cfg, tokenizer, device)
    metrics_log_path = os.path.join(run_cfg.dir, "metrics.jsonl")

    start_iter = 0
    saved_sched_state = None
    if run_cfg.load_checkpoint:
        start_iter, saved_sched_state = load_checkpoint_state(
            run_cfg.load_checkpoint, device, model, ema, opt,
        )

    if run_cfg.lr_decay:
        # Skip warmup on resume; decay from optimizer.lr to lr_min over the
        # remaining training budget. Don't replay the saved scheduler state.
        decay_steps = max(1, run_cfg.num_iters - start_iter)
        scheduler = CosineDecayScheduler(opt, total_steps=decay_steps, min_lr=run_cfg.lr_min)
    else:
        scheduler = WarmUpScheduler(opt, run_cfg.warmup_iters)
        if saved_sched_state is not None:
            scheduler.load_state_dict(saved_sched_state)

    model.train()

    training_iter = start_iter
    num_iters = run_cfg.num_iters
    log_rate = run_cfg.log_rate

    while training_iter < num_iters:
        pbar = tqdm(dataloader, total=len(dataloader), leave=False)
        for data_ in pbar:
            if training_iter >= num_iters:
                break

            for key, value in data_.items():
                data_[key] = value.to(device=device)

            if is_continuous_category:
                cat_bits = analog_bit.encode(data_["y"])  # [B, L, num_bits]
                data_ = {"x": torch.cat([data_["x"], cat_bits], dim=-1), "mask": data_["mask"]}

            opt.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if run_cfg.aux_l1_weight > 0 and not is_continuous_category:
                    def aux_l1_loss_fn(prediction, sample, x1, y1):
                        mask_t_shaped = sample.mask_t.unsqueeze(-1)
                        masked_positions = (sample.yt == interpolant.mask_token)
                        l1 = (sample.x1_ordered - prediction.clean_data).abs() * mask_t_shaped
                        l1[:, 0] = 0.0
                        l1 = l1.sum(dim=-1)[~masked_positions]
                        l1 = l1.mean() / x1.shape[-1]
                        return {"aux_l1_loss": l1}
                    losses = interpolant.compute_loss(model, data_, extra_loss_fn=aux_l1_loss_fn)
                else:
                    losses = interpolant.compute_loss(model, data_)

                loss = (
                    losses["dsm_loss"]
                    + losses["discrete_unmasking_loss"]
                    + losses["euclidean_unmasking_loss"]
                    + losses["insertion_loss"]
                )
                if "aux_l1_loss" in losses:
                    loss = loss + run_cfg.aux_l1_weight * losses["aux_l1_loss"]

            loss.backward()
            update_ema(ema, model, decay=run_cfg.ema_beta)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            for param in model.parameters():
                if param.grad is not None:
                    torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)

            opt.step()
            scheduler.step()
            training_iter += 1
            loss_val = loss.detach().item()

            pbar.set_description(
                f"Iter {training_iter} --- DSM: {losses['dsm_loss']:.4f}, "
                f"Disc: {losses['discrete_unmasking_loss']:.4f}, "
                f"Euc: {losses['euclidean_unmasking_loss']:.4f}, "
                f"Ins: {losses['insertion_loss']:.4f}"
            )

            if run_cfg.enable_wandb:
                log_dict = {"loss": loss_val, "step": training_iter, **{k: v.item() for k, v in losses.items()}}
                wandb.log(log_dict)

            if training_iter % log_rate == 0 or training_iter == num_iters:
                path = os.path.join(run_cfg.dir, f"itr_{training_iter}", "")
                os.makedirs(path, exist_ok=True)
                save_ckpt(model, ema, opt, scheduler, os.path.join(path, "snapshot.pt"))
                model.eval()

                vis_dir = os.path.join(path, "layouts")
                os.makedirs(vis_dir, exist_ok=True)

                if is_continuous_category:
                    n_vis = 20
                    idx = torch.randint(0, train_lengths.shape[0], (n_vis,))
                    lens = train_lengths[idx].to(device)
                    active = torch.arange(max_length, device=device).unsqueeze(0) < lens.unsqueeze(1)
                    out = interpolant.sampling(ema, 50, active)
                    geom, cat_bits = out[..., :4], out[..., 4:]
                    cat_ids = analog_bit.decode(cat_bits).clamp(0, tokenizer.vocab_size - 1).long()
                    for i in range(n_vis):
                        L = int(lens[i])
                        plot_layout_sample(
                            geom[i, :L].cpu(), cat_ids[i, :L].cpu(), active[i, :L].cpu(),
                            tokenizer, os.path.join(vis_dir, f"layout_{i}.png"),
                            iter_num=training_iter, title=f"iter {training_iter} | sample {i}",
                        )
                    model.train()
                    continue

                samples = interpolant.sampling(model, 50, 20, max_length + 1, device, return_trace=True)

                if evaluator is not None:
                    metrics = _run_eval(
                        evaluator, interpolant, ema, cfg.eval, max_length, device,
                        train_lengths=train_lengths,
                    )
                    metrics["iter"] = training_iter
                    with open(metrics_log_path, "a") as f:
                        f.write(json.dumps(metrics) + "\n")
                    print(
                        f"[eval @ iter {training_iter}] "
                        f"FID={metrics['fid']:.3f} "
                        f"Align={metrics['alignment-LayoutGAN++']:.4f} "
                        f"Overlap={metrics['overlap-LayoutGAN++']:.4f}"
                    )
                    if run_cfg.enable_wandb:
                        eval_log = {f"eval/{k}": v for k, v in metrics.items() if k != "iter"}
                        eval_log["step"] = training_iter
                        wandb.log(eval_log)

                for i, sample in enumerate(samples):
                    mask_t = sample.y_mask_t if hasattr(sample, "y_mask_t") else sample.mask_t
                    plot_layout_sample(
                        sample.xt.cpu(),
                        sample.yt.cpu(),
                        mask_t.cpu(),
                        tokenizer,
                        os.path.join(vis_dir, f"layout_{i}.png"),
                        iter_num=training_iter,
                        title=f"iter {training_iter} | sample {i}",
                    )

                model.train()

    save_ckpt(model, ema, opt, scheduler, os.path.join(run_cfg.dir, "final_checkpoint.pt"))

    if run_cfg.enable_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
