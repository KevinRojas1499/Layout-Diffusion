"""Hydra schema definitions for layout training config validation."""

from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from hydra.types import MISSING


@dataclass
class RunSchema:
    """Schema for run / training loop configuration."""

    seed: int = 42
    batch_size: int = 128
    num_iters: int = 5000
    log_rate: int = 500
    warmup_iters: int = 100
    num_workers: int = 2
    dir: str = "outputs/layout"
    load_checkpoint: str | None = None
    enable_wandb: bool = False
    run_name: str = ""
    ema_beta: float = 0.9999


@dataclass
class DatasetSchema:
    """Schema for dataset configuration."""

    name: str = "publaynet"
    data_path: str | None = None
    max_length: int = 64
    split: str = "train"
    max_samples: int | None = None


@dataclass
class ModelSchema:
    """Schema for model configuration."""

    name: str = "MMDiTBothVar"
    euclidean_dim: int = 4
    symbols_depth: int = 4
    positions_depth: int = 4
    depth: int = 4
    hidden_dim: int = 256


@dataclass
class OptimizerSchema:
    """Schema for optimizer configuration."""

    name: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.0
    muon_lr: float = 5e-4
    muon_weight_decay: float = 0.01


@dataclass
class InterpolantSchema:
    """Schema for interpolant configuration."""

    name: str = "multimodal_both"


@dataclass
class LayoutConfigSchema:
    """Root schema validating the full layout training config."""

    run: RunSchema = MISSING
    dataset: DatasetSchema = MISSING
    model: ModelSchema = MISSING
    optimizer: OptimizerSchema = MISSING
    interpolant: InterpolantSchema = MISSING


def register_configs() -> None:
    """
    Register schema configs in ConfigStore for validation.

    Hydra 1.1+ migration: Only register schemas with base_* names.
    YAML configs (run/default.yaml, dataset/publaynet.yaml, etc.) extend these
    via their defaults list. This avoids the deprecated "automatic schema matching"
    warning that occurs when ConfigStore and YAML share the same name.
    """
    cs = ConfigStore.instance()

    # Schemas only - YAML files extend these via defaults: [base_*]
    cs.store(group="run", name="base_run", node=RunSchema)
    cs.store(group="dataset", name="base_dataset", node=DatasetSchema)
    cs.store(group="model", name="base_model", node=ModelSchema)
    cs.store(group="optimizer", name="base_optimizer", node=OptimizerSchema)
    cs.store(group="interpolant", name="base_interpolant", node=InterpolantSchema)
