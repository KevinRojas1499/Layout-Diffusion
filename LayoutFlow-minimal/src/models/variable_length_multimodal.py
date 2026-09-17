import os
import torch
import lightning.pytorch as pl

from src.metrics import compute_alignment, compute_overlap
from src.visualize import draw_layout
from src.fid import FID_score
from src.variable_length_multimodal_interpolant import VariableLengthMultimodalInterpolant


class VariableLengthMultimodalModel(pl.LightningModule):
    '''
    Genuinely variable-length version of MultimodalMaskingModel: elements are
    inserted (born) over the course of generation rather than all existing
    from t=0 on a fixed canvas. See VariableLengthMultimodalInterpolant's
    docstring. Unlike the fixed-length models, `inference` doesn't take a
    batch's ground-truth active mask at all -- length is decided autonomously
    by the model's own insertion process, so `val_loss` (which needs a
    per-sample generated/ground-truth correspondence) isn't meaningful here
    and is dropped; FID/Alignment/Overlap don't need that correspondence.
    '''

    def __init__(
        self, backbone_model, optimizer, scheduler=None,
        scheduler_patience=10, scheduler_factor=0.1,
        pretrained_dir='./pretrained', dataset='RICO', num_cat=6,
        fid_calc_every_n=20, format='xywh', inference_steps=100, max_len=20, vis_dir=None,
    ):
        super().__init__()
        self.optimizer_partial = optimizer
        self.scheduler_setting = scheduler
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.dataset = dataset
        self.format = format
        self.num_cat = num_cat
        self.inference_steps = inference_steps
        self.max_len = max_len
        self.vis_dir = vis_dir
        self.fid_calc_every_n = fid_calc_every_n

        self.geom_dim = 4
        self.interpolant = VariableLengthMultimodalInterpolant(geom_dim=self.geom_dim, num_cat=num_cat)
        self.model = backbone_model
        self.fid_model = FID_score(dataset, pretrained_dir, calc_every_n=fid_calc_every_n) if fid_calc_every_n else None

        self.gen_data = {'bbox': [], 'label': [], 'pad_mask': []}
        self.fid_score = 0
        self.save_hyperparameters(ignore=['backbone_model'])

    def configure_optimizers(self):
        optimizer = self.optimizer_partial(params=self.model.parameters())
        if self.scheduler_setting == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=self.scheduler_patience, factor=self.scheduler_factor)
            return [optimizer], [{'scheduler': scheduler, 'monitor': 'FID_Layout',
                                   'frequency': self.fid_calc_every_n}]
        return optimizer

    def training_step(self, batch, batch_idx):
        x1 = 2 * batch['bbox'] - 1   # [-1, 1] rescale, matches flow matching's convention
        y1 = batch['type'].long()
        active = batch['mask'].squeeze(-1)
        losses = self.interpolant.compute_loss(self.model, {'x': x1, 'y': y1, 'mask': active})
        loss = sum(losses.values())
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log('geom_loss', losses['geom_loss'], prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log('cat_loss', losses['cat_loss'], prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log('insertion_loss', losses['insertion_loss'], prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def inference(self, batch_size, device):
        xt, yt, alive = self.interpolant.sampling(self.model, self.inference_steps, batch_size, self.max_len, device)
        geom = (xt[:, 1:] + 1) / 2   # drop BOS slot, undo [-1, 1] rescale
        cat = yt[:, 1:].clamp(0, self.num_cat - 1).long()
        pad_mask = alive[:, 1:]
        return geom, cat, pad_mask

    def validation_step(self, batch, batch_idx):
        B = batch['bbox'].shape[0]
        geom_pred, cat, pad_mask = self.inference(B, batch['bbox'].device)

        self.gen_data['bbox'].append(geom_pred)
        self.gen_data['label'].append(cat)
        self.gen_data['pad_mask'].append(pad_mask)

        if batch_idx == 0:
            self.log('FID_Layout', self.fid_score, on_epoch=True, sync_dist=True)
            if self.vis_dir:
                self.save_example_layouts(batch, geom_pred, cat, pad_mask)

    def on_validation_epoch_end(self):
        bbox = torch.cat(self.gen_data['bbox'])
        label = torch.cat(self.gen_data['label'])
        pad_mask = torch.cat(self.gen_data['pad_mask'])

        self.log_dict({'Alignment': compute_alignment(bbox.cpu(), pad_mask.cpu()) * 100})
        self.log_dict({'Overlap': compute_overlap(bbox.cpu(), pad_mask.cpu())})
        self.log_dict({'gen_mean_length': pad_mask.float().sum(dim=1).mean()})

        if self.fid_calc_every_n != 0:
            self.fid_score = self.fid_model.calc_FID(
                {'bbox': bbox, 'label': label, 'pad_mask': pad_mask}, format=self.format,
            )
            self.log_dict({'FID_Layout': self.fid_score})

        for key in self.gen_data:
            self.gen_data[key] = []

    def save_example_layouts(self, batch, geom_pred, cat, pad_mask, n=4):
        os.makedirs(self.vis_dir, exist_ok=True)
        for i in range(min(n, geom_pred.shape[0])):
            step = self.global_step
            L_true = int(batch['mask'][i].squeeze(-1).sum())
            L_gen = int(pad_mask[i].sum())
            draw_layout(batch['bbox'][i, :L_true].cpu(), batch['type'][i, :L_true].cpu(), num_colors=self.num_cat) \
                .save(f'{self.vis_dir}/step{step}_sample{i}_gt.png')
            draw_layout(geom_pred[i, :L_gen].cpu(), cat[i, :L_gen].cpu(), num_colors=self.num_cat) \
                .save(f'{self.vis_dir}/step{step}_sample{i}_pred.png')
