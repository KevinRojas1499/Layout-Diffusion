import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import rootutils
import torch

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from src.ema import EMA

torch.set_float32_matmul_precision('medium')
rootutils.setup_root(__file__, indicator='.git', pythonpath=True)


@hydra.main(version_base=None, config_path='conf', config_name='train.yaml')
def main(cfg: DictConfig):
    logger = WandbLogger(save_dir=cfg.wandb_dir, name=cfg.expname, project='LayoutFlow-minimal') \
        if cfg.enable_wandb else None

    ckpt_dir = f'{cfg.run_dir}/checkpoints'
    trainer = instantiate(
        cfg.trainer,
        callbacks=[c for c in [
            EMA(decay=cfg.ema_decay) if cfg.ema_decay else None,
            ModelCheckpoint(dirpath=ckpt_dir, filename='ckpt-{epoch:02d}-{' + cfg.ckpt_monitor + ':.3f}',
                             save_top_k=3, monitor=cfg.ckpt_monitor, mode='min', save_last=True),
            ModelCheckpoint(dirpath=ckpt_dir, filename='periodic-{epoch:04d}', every_n_epochs=cfg.ckpt_every_n_epochs,
                            save_top_k=-1, save_on_train_epoch_end=True) if cfg.ckpt_every_n_epochs else None,
            LearningRateMonitor(logging_interval='step'),
        ] if c is not None],
        logger=logger,
    )
    if logger is not None and trainer.global_rank == 0:
        logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))

    train_loader = instantiate(cfg.dataset)
    val_loader = instantiate(cfg.dataset, dataset={'split': 'validation'}, shuffle=False)
    model = instantiate(cfg.model, dataset=cfg.dataset_name, format=cfg.data.format,
                         vis_dir=f'{cfg.run_dir}/vis' if cfg.visualize else None)

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader,
                ckpt_path=cfg.ckpt_path)


if __name__ == '__main__':
    main()
