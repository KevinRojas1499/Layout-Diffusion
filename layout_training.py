"""Layout training with Hydra config."""

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
from branching_flows_interpolant import BranchingFlowsInterpolant
from custom_datasets.multimodal_math import EquationsDataset, ParenthesizedEquationsDataset
from custom_datasets.publaynet import PublayNetDataset, PUBLAYNET_VOCAB
from utils.tokenizer import VocabTokenizer
from utils.optimizers import WarmUpScheduler, CombinedOptimizer
from models.mmdit_qm9 import MMDiTQM9, MMDiTBothVar
from visualize_dataset import plot_sample, plot_sample_2, plot_layout_sample
from multimodal_interpolant_both_var import MultimodalInterpolantBoth

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

    if ds_cfg.name == "publaynet":
        tokenizer = VocabTokenizer(vocab=PUBLAYNET_VOCAB)
        dataset = PublayNetDataset(
            tokenizer,
            max_length=max_length,
            split=ds_cfg.split,
            max_samples=ds_cfg.max_samples,
        )
        euclidean_dim = 4  # bbox: x_center, y_center, w, h
        hidden_dim = cfg.model.hidden_dim
    elif ds_cfg.name == "equations":
        tokenizer = VocabTokenizer(vocab={"+", "-", "*", "=", ".", "(", ")"})
        dataset = EquationsDataset(
            tokenizer,
            max_length=max_length,
            data_path=ds_cfg.data_path or "data/equations_l5.jsonl",
        )
        euclidean_dim = 1
        hidden_dim = 256
    elif ds_cfg.name == "parenthesis":
        tokenizer = VocabTokenizer(vocab={"+", "-", "*", "=", ".", "(", ")"})
        dataset = ParenthesizedEquationsDataset(
            tokenizer,
            data_path=ds_cfg.data_path or "data/equations_l5.jsonl",
        )
        max_length = dataset.max_length
        euclidean_dim = 1
        hidden_dim = 256
    else:
        raise ValueError(f"Unknown dataset: {ds_cfg.name}")

    return dataset, tokenizer, euclidean_dim, hidden_dim, max_length


def _get_model(cfg: LayoutConfigSchema, vocab_size: int, euclidean_dim: int, hidden_dim: int, device: torch.device):
    """Build model from config."""
    m_cfg = cfg.model
    dim_modalities = [hidden_dim, hidden_dim]
    dim_conds = [hidden_dim, hidden_dim]

    if m_cfg.name == "MMDiTBothVar":
        model = MMDiTBothVar(
            euclidean_dim=euclidean_dim,
            vocab_size=vocab_size,
            symbols_depth=m_cfg.symbols_depth,
            positions_depth=m_cfg.positions_depth,
            depth=m_cfg.depth,
            dim_modalities=dim_modalities,
            dim_joint_attn=hidden_dim,
            dim_conds=dim_conds,
        )
    elif m_cfg.name == "DiT":
        model = MMDiTQM9(
            branching_flows=cfg.interpolant.name == "branching",
            euclidean_dim=euclidean_dim,
            vocab_size=vocab_size,
            symbols_depth=m_cfg.symbols_depth,
            positions_depth=m_cfg.positions_depth,
            depth=m_cfg.depth,
            dim_modalities=dim_modalities,
            dim_joint_attn=hidden_dim,
            dim_conds=dim_conds,
        )
    else:
        raise ValueError(f"Unknown model: {m_cfg.name}")

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

    if int_cfg.name == "multimodal":
        return MultimodalInterpolant(
            max_length=max_length,
            vocab_size=tokenizer.vocab_size,
            mask_token=tokenizer.mask_token_id,
            pad_token=tokenizer.pad_token_id,
            bos_token=tokenizer.bos_token_id,
            euclidean_dim=euclidean_dim,
        )
    elif int_cfg.name == "multimodal_both":
        return MultimodalInterpolantBoth(
            max_length=max_length,
            vocab_size=tokenizer.vocab_size,
            mask_token=tokenizer.mask_token_id,
            pad_token=tokenizer.pad_token_id,
            bos_token=tokenizer.bos_token_id,
            euclidean_dim=euclidean_dim,
        )
    elif int_cfg.name == "branching":
        return BranchingFlowsInterpolant(
            vocab_size=tokenizer.vocab_size,
            mask_token=tokenizer.mask_token_id,
            pad_token=tokenizer.pad_token_id,
            euclidean_dim=euclidean_dim,
        )
    raise ValueError(f"Unknown interpolant: {int_cfg.name}")


def load_checkpoint(
    load_path: str,
    device: torch.device,
    model: torch.nn.Module,
    ema: torch.nn.Module,
    opt: torch.optim.Optimizer,
    scheduler: WarmUpScheduler,
) -> int:
    """Load checkpoint and return start iteration."""
    snapshot = torch.load(load_path, weights_only=True)
    model.load_state_dict(snapshot["model"], strict=False)
    ema.load_state_dict(snapshot["ema"], strict=False)
    opt.load_state_dict(snapshot["optimizer"])
    scheduler.load_state_dict(snapshot["scheduler"])
    return scheduler.last_epoch


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

    dataloader = DataLoader(
        dataset,
        batch_size=run_cfg.batch_size,
        shuffle=True,
        num_workers=run_cfg.num_workers,
        drop_last=True,
    )

    if run_cfg.enable_wandb:
        init_wandb(cfg)

    model = _get_model(cfg, tokenizer.vocab_size, euclidean_dim, hidden_dim, device)
    ema = deepcopy(model)
    opt = _get_optimizer(cfg, model)
    scheduler = WarmUpScheduler(opt, run_cfg.warmup_iters)
    scaler = torch.amp.GradScaler()
    interpolant = _get_interpolant(cfg, dataset, tokenizer, euclidean_dim)

    start_iter = 0
    if run_cfg.load_checkpoint:
        start_iter = load_checkpoint(
            run_cfg.load_checkpoint,
            device,
            model,
            ema,
            opt,
            scheduler,
        )

    model.train()

    training_iter = start_iter
    num_iters = run_cfg.num_iters
    log_rate = run_cfg.log_rate
    int_name = cfg.interpolant.name

    while training_iter < num_iters:
        pbar = tqdm(dataloader, total=len(dataloader), leave=False)
        for data_ in pbar:
            if training_iter >= num_iters:
                break

            for key, value in data_.items():
                data_[key] = value.to(device=device)

            opt.zero_grad()
            losses = interpolant.compute_loss(model, data_)

            if int_name == "multimodal":
                loss = (
                    losses["dsm_loss"]
                    + losses["discrete_unmasking_loss"]
                    + losses["euclidean_unmasking_loss"]
                    + losses["insertion_loss"]
                )
            elif int_name == "multimodal_both":
                loss = (
                    losses["dsm_loss"]
                    + losses["discrete_unmasking_loss"]
                    + losses["euc_insertion_loss"]
                    + losses["disc_insertion_loss"]
                )
            elif int_name == "branching":
                loss = (
                    losses["dsm_loss"]
                    + losses["discrete_unmasking_loss"]
                    + losses["insertion_loss"]
                )
            else:
                raise ValueError(f"Unknown interpolant: {int_name}")

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            update_ema(ema, model, decay=run_cfg.ema_beta)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            for param in model.parameters():
                if param.grad is not None:
                    torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)

            scaler.step(opt)
            scaler.update()
            scheduler.step()
            training_iter += 1
            loss_val = loss.detach().item()

            if int_name == "multimodal":
                pbar.set_description(
                    f"Iter {training_iter} --- DSM: {losses['dsm_loss']:.4f}, "
                    f"Disc: {losses['discrete_unmasking_loss']:.4f}, "
                    f"Euc: {losses['euclidean_unmasking_loss']:.4f}, "
                    f"Ins: {losses['insertion_loss']:.4f}"
                )
            elif int_name == "multimodal_both":
                pbar.set_description(
                    f"Iter {training_iter} --- DSM: {losses['dsm_loss']:.4f}, "
                    f"Disc: {losses['discrete_unmasking_loss']:.4f}, "
                    f"EucIns: {losses['euc_insertion_loss']:.4f}, "
                    f"DiscIns: {losses['disc_insertion_loss']:.4f}"
                )
            elif int_name == "branching":
                pbar.set_description(
                    f"Iter {training_iter} --- DSM: {losses['dsm_loss']:.4f}, "
                    f"Disc: {losses['discrete_unmasking_loss']:.4f}, "
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

                samples = interpolant.sampling(model, 50, 20, max_length + 1, device, return_trace=True)
                if cfg.dataset.name == "publaynet":
                    vis_dir = os.path.join(path, "layouts")
                    os.makedirs(vis_dir, exist_ok=True)
                for i, sample in enumerate(samples):
                    if cfg.dataset.name == "publaynet":
                        mask_t = getattr(sample, "y_mask_t", sample.mask_t)
                        plot_layout_sample(
                            sample.xt.cpu(),
                            sample.yt.cpu(),
                            mask_t.cpu(),
                            tokenizer,
                            os.path.join(vis_dir, f"layout_{i}.png"),
                            iter_num=training_iter,
                            title=f"iter {training_iter} | sample {i}",
                        )
                    elif int_name == "multimodal":
                        plot_sample(
                            sample.xt.cpu(),
                            sample.yt.cpu(),
                            sample.mask_t.cpu(),
                            os.path.join(path, f"sample_{i}.png"),
                            tokenizer,
                        )
                    elif int_name == "multimodal_both":
                        plot_sample_2(
                            sample.xt.cpu(),
                            sample.yt.cpu(),
                            sample.x_mask_t.cpu(),
                            sample.y_mask_t.cpu(),
                            os.path.join(path, f"sample_{i}.png"),
                            tokenizer,
                        )

                model.train()

    save_ckpt(model, ema, opt, scheduler, os.path.join(run_cfg.dir, "final_checkpoint.pt"))

    if run_cfg.enable_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
