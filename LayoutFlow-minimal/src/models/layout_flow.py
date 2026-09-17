import torch
import torch.nn as nn
import numpy as np
from torchcfm import ConditionalFlowMatcher
from torchdyn.core import NeuralODE

from src.models.base import BaseGenModel
from src.fid import FID_score
from src.analog_bit import AnalogBit


class LayoutFlow(BaseGenModel):
    '''
    Conditional flow matching over (bbox, AnalogBit-category) per layout element.
    Trimmed from upstream: dropped the 'discrete'/'continuous' category encodings
    (LayoutFlow's own config always uses AnalogBit), the alternate `cond_wrapper`
    inference path (an appendix-only ablation, never used by the main recipe), and
    the `refinement` task's truncated ODE time span (dropped along with the
    standalone conditional eval tasks -- see base.py's docstring). `train_traj` and
    `time_sampling` keep all three upstream options since they're cheap and are
    exactly the kind of knob you'd want to sweep when debugging a numbers gap.
    '''

    def __init__(
        self, backbone_model, sampler, optimizer, scheduler=None, loss_fcn='mse',
        pretrained_dir='./pretrained', format='xywh', sigma=0.0, fid_calc_every_n=20,
        dataset='RICO', num_cat=6, time_sampling='uniform', inference_steps=100,
        ode_solver='euler', cond='uncond', train_traj='linear', add_loss='',
        add_loss_weight=1, cf_guidance=0, vis_dir=None,
    ):
        self.format = format
        self.time_sampling = time_sampling
        self.time_weight = float(time_sampling.split('_')[-1]) if 'late_focus_' in time_sampling else 0.3
        self.fid_calc_every_n = fid_calc_every_n
        self.cond = cond
        self.analog_bit = AnalogBit(num_cat)
        fid_model = FID_score(dataset, pretrained_dir, calc_every_n=fid_calc_every_n) if fid_calc_every_n else None
        super().__init__(optimizer=optimizer, scheduler=scheduler, dataset=dataset, fid_model=fid_model,
                          vis_dir=vis_dir)

        self.attr_dim = int(np.ceil(np.log2(num_cat)))
        self.num_cat = num_cat
        self.model = backbone_model
        self.inference_steps = inference_steps
        self.ode_solver = ode_solver
        self.sampler = sampler
        self.cf_guidance = cf_guidance

        self.train_traj = train_traj
        self.FM = ConditionalFlowMatcher(sigma=sigma)
        self.loss_fcn = nn.MSELoss() if loss_fcn != 'l1' else nn.L1Loss()
        self.add_loss = add_loss
        self.add_loss_weight = add_loss_weight
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        cond_mask = self.get_cond_mask(batch)
        x0, x1 = self.get_start_end(batch)
        t = self.sample_t(x0)
        xt, ut = self.sample_xt(x0, x1, cond_mask, t)

        vt = self(xt, cond_mask, t.squeeze(-1))
        loss = self.loss_fcn(cond_mask * vt, cond_mask * ut)
        if self.add_loss:
            loss = self.additional_losses(cond_mask, ut, vt, loss)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def sample_xt(self, x0, x1, cond_mask, t):
        eps = self.FM.sample_noise_like(x0) if self.FM.sigma != 0 else 0
        tpad = t.reshape(-1, *([1] * (x0.dim() - 1)))
        if self.train_traj == 'sin':
            xt = (1 - torch.sin(tpad * torch.pi / 2)) * x0 + torch.sin(tpad * torch.pi / 2) * x1 + self.FM.sigma * eps
            ut = torch.pi / 2 * torch.cos(tpad * torch.pi / 2) * (x1 - x0)
        elif self.train_traj == 'sincos':
            xt = torch.cos(tpad * torch.pi / 2) * x0 + torch.sin(tpad * torch.pi / 2) * x1 + self.FM.sigma * eps
            ut = torch.pi / 2 * (torch.cos(tpad * torch.pi / 2) * x1 - torch.sin(tpad * torch.pi / 2) * x0)
        else:  # linear
            xt = self.FM.sample_xt(x0, x1, t, eps)
            ut = self.FM.compute_conditional_flow(x0, x1, t, xt)
        xt = (1 - cond_mask) * x1 + cond_mask * xt
        return xt, ut

    def inference(self, batch, full_traj=False):
        x0 = self.sampler.sample(batch)
        cond_mask = self.get_cond_mask(batch)

        conv_type = self.analog_bit.encode(batch['type'])
        ref = torch.cat([batch['bbox'], conv_type], dim=-1)
        cond_x = self.sampler.preprocess(ref)
        cond_x = batch['mask'] * cond_x + (~batch['mask']) * ref

        vector_field = FlowWrapper(self, cond_x, cond_mask, cf_guidance=self.cf_guidance)
        node = NeuralODE(vector_field, solver=self.ode_solver, sensitivity='adjoint', atol=1e-4, rtol=1e-4)
        traj = node.trajectory(x0, t_span=torch.linspace(0, 1, self.inference_steps))
        traj = self.sampler.preprocess(traj, reverse=True)

        cont_cat = self.analog_bit.decode(traj[..., self.geom_dim:])
        cont_cat = (1 - cond_mask[:, :, -1]) * batch['type'] + cond_mask[:, :, -1] * cont_cat
        cat = torch.clip(cont_cat.to(torch.int), 0, self.num_cat - 1)
        traj = (1 - cond_mask[:, :, :self.geom_dim]) * batch['bbox'][None] \
            + cond_mask[:, :, :self.geom_dim] * traj[..., :self.geom_dim]

        return (traj, cat, cont_cat) if full_traj else (traj[-1], cat[-1])


class FlowWrapper(nn.Module):
    '''Wraps the backbone into torchdyn's expected (t, x) -> dx/dt signature, holding
    conditioned-on entries fixed to their ground-truth value at every ODE step, with
    optional classifier-free guidance (cf_guidance=0 disables it, matching upstream default).'''

    def __init__(self, model, cond_x, cond_mask, cf_guidance=0):
        super().__init__()
        self.model = model
        self.cond_x = cond_x
        self.cond_mask = cond_mask
        self.cf_guidance = cf_guidance
        if cf_guidance:
            self.uncond_mask = torch.ones_like(cond_mask)

    def forward(self, t, x, *args, **kwargs):
        x = (1 - self.cond_mask) * self.cond_x + self.cond_mask * x
        v = self.model(x, self.cond_mask, t.repeat(x.shape[0]))
        if self.cf_guidance:
            v = (1 + self.cf_guidance) * v - self.cf_guidance * self.model(x, self.uncond_mask, t.repeat(x.shape[0]))
        return v
