import os
import torch
import lightning.pytorch as pl

from src.metrics import compute_alignment, compute_overlap
from src.visualize import draw_layout
from src.fid import FID_score
from src.multimodal_masking_interpolant import MultimodalMaskingInterpolant


class MultimodalMaskingModel(pl.LightningModule):
    '''
    Same Lightning-wrapper style as ContinuousMaskingModel, but built on
    MultimodalMaskingInterpolant: geometry stays continuous, category is
    handled as a genuine discrete channel (real [MASK] id + cross-entropy)
    instead of analog-bit regression. See that interpolant's docstring for why.
    '''

    def __init__(
        self, backbone_model, optimizer, scheduler=None,
        scheduler_patience=10, scheduler_factor=0.1, cat_loss_weight=1.0,
        pretrained_dir='./pretrained', dataset='RICO', num_cat=6,
        fid_calc_every_n=20, format='xywh', inference_steps=100, vis_dir=None,
    ):
        super().__init__()
        self.optimizer_partial = optimizer
        self.scheduler_setting = scheduler
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.cat_loss_weight = cat_loss_weight
        self.dataset = dataset
        self.format = format
        self.num_cat = num_cat
        self.inference_steps = inference_steps
        self.vis_dir = vis_dir
        self.fid_calc_every_n = fid_calc_every_n

        self.geom_dim = 4
        self.interpolant = MultimodalMaskingInterpolant(geom_dim=self.geom_dim, num_cat=num_cat)
        self.model = backbone_model
        self.fid_model = FID_score(dataset, pretrained_dir, calc_every_n=fid_calc_every_n) if fid_calc_every_n else None

        self.gen_data = {'bbox': [], 'label': [], 'pad_mask': []}
        self.fid_score = 0
        self.save_hyperparameters(ignore=['backbone_model'])

    def configure_optimizers(self):
        optimizer = self.optimizer_partial(params=self.model.parameters(), betas=(0.9, 0.98))
        if self.scheduler_setting == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=self.scheduler_patience, factor=self.scheduler_factor)
            return [optimizer], [{'scheduler': scheduler, 'monitor': 'FID_Layout',
                                   'frequency': self.fid_calc_every_n}]
        return optimizer

    def _make_geom_x1(self, batch):
        # [-1, 1] rescale to match the N(0, 1) corruption noise's scale (same
        # fix as ContinuousMaskingModel -- see git log).
        return 2 * batch['bbox'] - 1

    def training_step(self, batch, batch_idx):
        x1 = self._make_geom_x1(batch)
        y1 = batch['type'].long()
        active = batch['mask'].squeeze(-1)
        losses = self.interpolant.compute_loss(self.model, {'x': x1, 'y': y1, 'mask': active})
        loss = losses['geom_loss'] + self.cat_loss_weight * losses['cat_loss']
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log('geom_loss', losses['geom_loss'], prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log('cat_loss', losses['cat_loss'], prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def inference(self, batch):
        active = batch['mask'].squeeze(-1)
        xt, yt = self.interpolant.sampling(self.model, self.inference_steps, active)
        geom = (xt + 1) / 2
        cat = yt.clamp(0, self.num_cat - 1).long()
        return geom, cat

    def validation_step(self, batch, batch_idx):
        geom_pred, cat = self.inference(batch)
        pad_mask = batch['mask'].squeeze(-1)
        val_loss = ((geom_pred - batch['bbox']) ** 2 * pad_mask.unsqueeze(-1)).sum() / pad_mask.sum().clamp(min=1)
        self.log('val_loss', val_loss, sync_dist=True)

        self.gen_data['bbox'].append(geom_pred)
        self.gen_data['label'].append(cat)
        self.gen_data['pad_mask'].append(pad_mask)

        if batch_idx == 0:
            self.log('FID_Layout', self.fid_score, on_epoch=True, sync_dist=True)
            if self.vis_dir:
                self.save_example_layouts(batch, geom_pred, cat)

    def on_validation_epoch_end(self):
        bbox = torch.cat(self.gen_data['bbox'])
        label = torch.cat(self.gen_data['label'])
        pad_mask = torch.cat(self.gen_data['pad_mask'])

        self.log_dict({'Alignment': compute_alignment(bbox.cpu(), pad_mask.cpu()) * 100})
        self.log_dict({'Overlap': compute_overlap(bbox.cpu(), pad_mask.cpu())})

        if self.fid_calc_every_n != 0:
            self.fid_score = self.fid_model.calc_FID(
                {'bbox': bbox, 'label': label, 'pad_mask': pad_mask}, format=self.format,
            )
            self.log_dict({'FID_Layout': self.fid_score})

        for key in self.gen_data:
            self.gen_data[key] = []

    def save_example_layouts(self, batch, geom_pred, cat, n=4):
        os.makedirs(self.vis_dir, exist_ok=True)
        for i in range(min(n, batch['bbox'].shape[0])):
            L = int(batch['mask'][i].squeeze(-1).sum())
            step = self.global_step
            draw_layout(batch['bbox'][i, :L].cpu(), batch['type'][i, :L].cpu(), num_colors=self.num_cat) \
                .save(f'{self.vis_dir}/step{step}_sample{i}_gt.png')
            draw_layout(geom_pred[i, :L].cpu(), cat[i, :L].cpu(), num_colors=self.num_cat) \
                .save(f'{self.vis_dir}/step{step}_sample{i}_pred.png')
