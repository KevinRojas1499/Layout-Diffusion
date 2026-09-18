import os
import torch
import lightning.pytorch as pl

from src.metrics import compute_alignment, compute_overlap
from src.utils import convert_bbox
from src.visualize import draw_layout


class BaseGenModel(pl.LightningModule):
    '''
    Shared training-loop plumbing (optimizer, conditioning masks, validation/FID
    loop) around a swappable flow/diffusion model. Trimmed from upstream's
    BaseGenModel: dropped the wandb/tensorboard-dual-path image+trajectory
    visualization (kept a much simpler periodic PNG dump instead), the
    conditional-generation-task branches of `get_cond_mask` (cat_cond/size_cond/
    elem_compl/refinement as *standalone* eval tasks -- the `random4` branch below
    still trains on a mix of all of them, that's core to LayoutFlow's method, just
    the ability to *evaluate* on them alone as a separate task was dropped along
    with test.py's task menu), and the dual uncond+cond validation FID (only uncond
    is used to drive the LR scheduler, so only uncond is computed here).
    '''

    def __init__(self, optimizer, scheduler=None, dataset='', fid_model=None, vis_dir=None):
        super().__init__()
        self.optimizer_partial = optimizer
        self.scheduler_setting = scheduler
        self.dataset = dataset
        self.fid_model = fid_model
        self.vis_dir = vis_dir
        self.geom_dim = 4

        self.gen_data = {'bbox': [], 'label': [], 'pad_mask': []}
        self.fid_score = 0

    def configure_optimizers(self):
        optimizer = self.optimizer_partial(params=self.model.parameters(), betas=(0.9, 0.98))
        if self.scheduler_setting == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
            return [optimizer], [{'scheduler': scheduler, 'monitor': 'FID_Layout',
                                   'frequency': self.fid_model.calc_every_n}]
        return optimizer

    def forward(self, xt, mask_cond, t):
        geom, attr = xt[..., :self.geom_dim], xt[..., self.geom_dim:]
        return self.model(geom, attr, mask_cond, t)

    def get_start_end(self, batch):
        conv_type = self.analog_bit.encode(batch['type'])
        gt = torch.cat([batch['bbox'], conv_type], dim=-1)
        x0 = self.sampler.sample(batch)
        x1 = self.sampler.preprocess(gt)
        x1 = batch['mask'] * x1 + (~batch['mask']) * gt
        return x0, x1

    def sample_t(self, x0):
        return torch.rand(x0.shape[0]).type_as(x0)

    def get_cond_mask(self, batch):
        '''cond_mask: 1 = free (predicted), 0 = held fixed (given).'''
        shape = (*batch['bbox'].shape[:2], self.geom_dim + self.attr_dim)
        if self.cond == 'uncond':
            return torch.ones(shape, dtype=torch.int, device=self.device)
        assert 'random4' in self.cond, f'unsupported cond mode: {self.cond}'
        # Mix four conditioning regimes within one batch so the model learns to
        # handle each at inference: uncond / element-completion / cat_cond / size_cond.
        cond_mask = torch.ones(shape, dtype=torch.int, device=self.device)
        div = batch['bbox'].shape[0] // 4
        idx_b, idx_s = [], []
        for i, l in enumerate(batch['length'][:div]):
            num_elem = l * 0.2 * torch.rand(1).to(self.device)
            if l > 1:
                idx = torch.multinomial(torch.arange(l).float(), int(num_elem.item()) + 1).tolist()
                idx_b += [i] * len(idx)
                idx_s += idx
        cond_mask[torch.tensor(idx_b), torch.tensor(idx_s)] = 0
        cond_mask[div:2 * div, :, self.geom_dim:] = 0     # cat_cond: category given
        cond_mask[2 * div:3 * div, :, 2:] = 0              # size_cond: category+size given
        return cond_mask

    def loss(self, pred, gt, seq_len):
        total = 0
        for i in range(pred.shape[0]):
            L = seq_len[i]
            total += self.loss_fcn(pred[i, :L], gt[i, :L])
        return total / pred.shape[0]

    def additional_losses(self, cond_mask, ut, vt, loss):
        self.log('flow_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        assert self.add_loss == 'geom_l1_loss'
        add_loss = torch.nn.functional.l1_loss(
            cond_mask[..., :self.geom_dim] * vt[..., :self.geom_dim],
            cond_mask[..., :self.geom_dim] * ut[..., :self.geom_dim],
        )
        self.log(self.add_loss, add_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss + self.add_loss_weight * add_loss

    def validation_step(self, batch, batch_idx):
        # Upstream evaluates unconditionally when trained on a random cond mix;
        # without this switch 3/4 of every val batch is handed ground truth and
        # FID_Layout reads ~0.5 too low (1.7 vs 2.1-2.4 on the RICO SOTA ckpt).
        train_cond, self.cond = self.cond, 'uncond'
        geom_pred, cat, *gen_mask = self.inference(batch)
        self.cond = train_cond
        loss = self.loss(geom_pred, batch['bbox'], batch['length'])
        self.log('val_loss', loss, sync_dist=True)

        if gen_mask:
            # variable-length models generate their own length (front-packed mask);
            # drop the (rare) empty layouts, which the FID net cannot embed
            pad_mask = gen_mask[0]
            self.log('gen_len_mean', pad_mask.sum(1).float().mean(), sync_dist=True)
            self.log('gen_len_abs_err', (pad_mask.sum(1).float().mean() - batch['length'].float().mean()).abs(),
                     sync_dist=True)
            keep = pad_mask.any(1)
            geom_pred, cat, pad_mask = geom_pred[keep], cat[keep], pad_mask[keep]
        else:
            pad_mask = torch.zeros(geom_pred.shape[:2], device=self.device, dtype=bool)
            for i, L in enumerate(batch['length']):
                pad_mask[i, :L] = True
        self.gen_data['bbox'].append(geom_pred)
        self.gen_data['label'].append(cat)
        self.gen_data['pad_mask'].append(pad_mask)

        if batch_idx == 0:
            self.log('FID_Layout', self.fid_score, on_epoch=True, sync_dist=True)
            if self.vis_dir and not gen_mask:
                self.save_example_layouts(batch, geom_pred, cat)

    def on_validation_epoch_end(self):
        bbox = torch.cat(self.gen_data['bbox'])
        label = torch.cat(self.gen_data['label'])
        pad_mask = torch.cat(self.gen_data['pad_mask'])

        self.log_dict({'Alignment': compute_alignment(bbox.cpu(), pad_mask.cpu()) * 100})
        self.log_dict({'Overlap': compute_overlap(bbox.cpu(), pad_mask.cpu())})

        if self.fid_calc_every_n != 0:
            self.fid_score = self.fid_model.calc_FID({'bbox': bbox, 'label': label, 'pad_mask': pad_mask},
                                                       format=self.format)
            self.log_dict({'FID_Layout': self.fid_score})

        for key in self.gen_data:
            self.gen_data[key] = []

    def save_example_layouts(self, batch, geom_pred, cat, n=4):
        os.makedirs(self.vis_dir, exist_ok=True)
        for i in range(min(n, len(batch['length']))):
            L = batch['length'][i]
            step = self.global_step
            draw_layout(batch['bbox'][i, :L].cpu(), batch['type'][i, :L].cpu(), num_colors=self.num_cat) \
                .save(f'{self.vis_dir}/step{step}_sample{i}_gt.png')
            draw_layout(geom_pred[i, :L].cpu(), cat[i, :L].cpu(), num_colors=self.num_cat) \
                .save(f'{self.vis_dir}/step{step}_sample{i}_pred.png')
