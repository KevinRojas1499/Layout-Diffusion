'''
Edit Flows (Havasi, Karrer, Gat, Chen 2025) on LayoutDM-tokenized layouts: the strictly discrete,
token-level baseline for variable-length generation. Follows the reference constructions of
- LayoutDM's LayoutSequenceTokenizer (CyberAgentAILab/layout-dm): a layout is the token sequence
  (c_1, x_1, y_1, w_1, h_1, c_2, ...) with categories first in the vocabulary, then one shared
  32-bin vocabulary for the four linearly quantized coordinates, in the dataset's element order;
- the educational Edit Flows implementation (TheMatrixMaster/edit-flows-demo): empty source, a
  gap-token alignment in which every target token is a pending insertion, kappa-interpolated
  conditional path (each target token present at time t w.p. kappa(t)), a transformer with learned
  positions + BOS that outputs per-position insert/substitute/delete rates and insert/substitute
  token distributions, the Bregman loss  sum(rates) - sum_{pending edits} log u(edit) kappa'/(1-kappa),
  and Euler sampling with an adaptive step (eq. 23 and Alg. 1 of the paper).
Nothing from LayoutFlowVarLen is used (no continuous flow, no GMM, no set structure): partial
elements exist mid-trajectory as they would in text, and decoding discards invalid 5-token groups
the way LayoutDM does.
'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import BaseGenModel
from src.fid import FID_score


class LayoutDMTokenizer:
    '''c-x-y-w-h, shared linear 32-bin box vocabulary, LayoutDM conventions (boxes are cx, cy, w, h in [0,1]).'''

    def __init__(self, num_cat, num_bins=32):
        self.C = num_cat - 1                 # dataset categories are 1..num_cat-1 (0 = pad id)
        self.nb = num_bins
        self.V = self.C + num_bins           # regular vocabulary
        self.BOS, self.PAD = self.V, self.V + 1
        self.vocab = self.V + 2

    def encode(self, bbox, label, mask):
        '''bbox (B,S,4) cxcywh in [0,1], label (B,S) 1-indexed, mask (B,S) -> tokens (B,S*5) with PAD, per-token mask.'''
        d = 1 / self.nb
        q = bbox.clone()
        q[..., :2] = bbox[..., :2].clamp(0.0, 1.0 - d)
        q[..., 2:] = bbox[..., 2:].clamp(d, 1.0) - d
        q = (self.nb * q).round().long().clamp(0, self.nb - 1) + self.C
        tok = torch.cat([(label - 1).clamp(min=0).unsqueeze(-1), q], -1)                  # (B,S,5)
        tok = torch.where(mask.unsqueeze(-1), tok, torch.full_like(tok, self.PAD))
        return tok.flatten(1), mask.unsqueeze(-1).expand(-1, -1, 5).flatten(1)

    def decode(self, tokens, max_len):
        '''tokens (B,L) with PAD -> bbox (B,max_len,4), label (B,max_len) 1-indexed, pad_mask (B,max_len).'''
        B, L = tokens.shape
        dev = tokens.device
        bbox = torch.zeros(B, max_len, 4, device=dev); label = torch.zeros(B, max_len, dtype=torch.long, device=dev)
        keep = torch.zeros(B, max_len, dtype=torch.bool, device=dev)
        d = 1 / self.nb
        for b in range(B):
            seq = tokens[b][tokens[b] != self.PAD]
            seq = seq[: (len(seq) // 5) * 5].view(-1, 5)                                   # a trailing partial element is dropped
            ok = (seq[:, 0] < self.C) & (seq[:, 1:] >= self.C).all(1) & (seq[:, 1:] < self.V).all(1)
            seq = seq[ok][:max_len]                                                         # LayoutDM: corrupted groups are discarded
            n = len(seq)
            if n:
                q = (seq[:, 1:] - self.C).float()
                box = torch.stack([q[:, 0] / self.nb, q[:, 1] / self.nb, (q[:, 2] + 1) / self.nb, (q[:, 3] + 1) / self.nb], -1)
                bbox[b, :n], label[b, :n], keep[b, :n] = box, seq[:, 0] + 1, True
        return bbox, label, keep


class EditFlowTransformer(nn.Module):
    '''The demo's SimpleEditFlowsTransformer: tokens + learned positions + time, per-position rates and token heads.'''

    def __init__(self, vocab, max_len, d_model=512, nhead=8, num_layers=4, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len + 1, d_model)
        self.time = nn.Sequential(nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True, norm_first=True)
        self.layers = nn.TransformerEncoder(layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.rates = nn.Linear(d_model, 3)
        self.ins_logits = nn.Linear(d_model, vocab)
        self.sub_logits = nn.Linear(d_model, vocab)

    def forward(self, tokens, t, pad_mask):
        '''tokens (B,L) incl. BOS at 0, t (B,), pad_mask (B,L) True = pad -> rates (B,L,3) [ins, sub, del], ins_logp, sub_logp (B,L,V)'''
        B, L = tokens.shape
        x = self.tok(tokens) + self.pos(torch.arange(L, device=tokens.device)).unsqueeze(0) + self.time(t.view(-1, 1)).unsqueeze(1)
        h = self.norm(self.layers(x, src_key_padding_mask=pad_mask))
        live = (~pad_mask).unsqueeze(-1).float()
        rates = F.softplus(self.rates(h)) * live
        return rates, F.log_softmax(self.ins_logits(h), -1), F.log_softmax(self.sub_logits(h), -1)


class LayoutEditFlow(BaseGenModel):
    def __init__(self, backbone_model=None, optimizer=None, scheduler=None, pretrained_dir='./pretrained', format='xywh',
                 fid_calc_every_n=20, dataset='RICO', num_cat=6, max_len=20, num_bins=32, inference_steps=100,
                 d_model=512, nhead=8, num_layers=4, dim_feedforward=2048, dropout=0.1, ralf_cache=None, vis_dir=None):
        self.format = format
        self.fid_calc_every_n = fid_calc_every_n
        self.cond = 'uncond'
        if not fid_calc_every_n:
            fid_model = None
        elif ralf_cache:
            from src.fid_ralf import RalfFID
            fid_model = RalfFID(dataset, ralf_cache, calc_every_n=fid_calc_every_n)
        else:
            fid_model = FID_score(dataset, pretrained_dir, calc_every_n=fid_calc_every_n)
        super().__init__(optimizer=optimizer, scheduler=scheduler, dataset=dataset, fid_model=fid_model, vis_dir=vis_dir)
        self.num_cat, self.max_len, self.inference_steps = num_cat, max_len, inference_steps
        self.tokenizer = LayoutDMTokenizer(num_cat, num_bins)
        self.max_tokens = 5 * max_len
        self.loss_fcn = nn.MSELoss()        # only for base.validation_step's val_loss diagnostic
        self.model = EditFlowTransformer(self.tokenizer.vocab, self.max_tokens, d_model, nhead, num_layers, dim_feedforward, dropout)
        self.save_hyperparameters(ignore=['backbone_model'])

    # linear kappa(t) = t (the demo's CubicScheduler(a=1, b=1) reduces to it): hazard kappa' / (1 - kappa) = 1 / (1 - t)
    @staticmethod
    def kappa(t):
        return t

    def _with_bos(self, seq, mask):
        B = seq.shape[0]
        bos = torch.full((B, 1), self.tokenizer.BOS, dtype=torch.long, device=seq.device)
        return torch.cat([bos, seq], 1), torch.cat([torch.ones(B, 1, dtype=torch.bool, device=seq.device), mask], 1)

    def training_step(self, batch, batch_idx):
        tk = self.tokenizer
        x1, m1 = tk.encode(self.sampler_preprocess(batch['bbox']), batch['type'].long(), batch['mask'].squeeze(-1))
        B, L1 = x1.shape
        t = torch.rand(B, device=self.device)
        # conditional path from the empty source: every target token is a pending insertion, present w.p. kappa(t)
        present = m1 & (torch.rand(B, L1, device=self.device) < self.kappa(t).view(-1, 1))
        # x_t = the present tokens, packed; each absent target token is an insertion after the last present token before it
        order = torch.argsort((~present).int(), dim=1, stable=True)            # present first, in order
        n_t = present.sum(1)
        x_t = torch.gather(x1, 1, order)
        x_t = torch.where(torch.arange(L1, device=self.device).unsqueeze(0) < n_t.unsqueeze(1), x_t, torch.full_like(x_t, tk.PAD))
        xt_pad = x_t == tk.PAD
        x_t, xt_mask = self._with_bos(x_t, ~xt_pad)
        rates, ins_logp, sub_logp = self.model(x_t, t, ~xt_mask)               # positions 0..n_t (0 = BOS)
        # target insertions: token x1[j] pending at position (number of present tokens before j) in x_t (BOS = 0)
        pending = m1 & ~present
        pos = present.int().cumsum(1) - present.int()                            # present tokens strictly before j
        coeff = (1.0 / (1 - t)).clamp(max=1e3)                                    # kappa' / (1 - kappa), linear kappa
        b_idx = torch.arange(B, device=self.device).unsqueeze(1).expand(B, L1)[pending]
        log_u = rates[b_idx, pos[pending], 0].clamp(min=1e-20).log() + ins_logp[b_idx, pos[pending], x1[pending]]
        u_tot = rates.sum((1, 2))
        loss = (u_tot - torch.zeros(B, device=self.device).index_add_(0, b_idx, log_u) * coeff).mean()
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log('u_ins', (rates[..., 0].sum(1)).mean(), on_step=False, on_epoch=True, sync_dist=True)
        self.log('u_del_sub', (rates[..., 1:].sum((1, 2))).mean(), on_step=False, on_epoch=True, sync_dist=True)
        self.log('ins_mae', (rates[..., 0].sum(1) * (1 - t) - pending.sum(1).float()).abs().mean(), on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def sampler_preprocess(self, bbox):
        return bbox        # tokens are built from the raw cx,cy,w,h in [0,1]

    @torch.no_grad()
    def inference(self, batch):
        '''Alg. 1 with the demo's adaptive Euler step; starts from the empty sequence. Returns (bbox, label, pad_mask).'''
        tk = self.tokenizer
        B = batch['type'].shape[0]
        dev = self.device
        Lmax = self.max_tokens
        x = torch.full((B, Lmax), tk.PAD, dtype=torch.long, device=dev)
        n = torch.zeros(B, dtype=torch.long, device=dev)
        h0 = 1.0 / self.inference_steps
        t = 0.0
        while t < 1.0 - 1e-9:
            h = min(h0, (1 - t))                                                   # adaptive step: min(h, (1-kappa)/kappa')
            xb, mb = self._with_bos(x, torch.arange(Lmax, device=dev).unsqueeze(0) < n.unsqueeze(1))
            rates, ins_logp, sub_logp = self.model(xb, torch.full((B,), t, device=dev), ~mb)
            lam_ins, lam_sub, lam_del = rates.unbind(-1)
            p_ins = 1 - torch.exp(-h * lam_ins)
            p_ds = 1 - torch.exp(-h * (lam_sub + lam_del))
            ins = (torch.rand_like(p_ins) < p_ins) & mb
            ds = (torch.rand_like(p_ds) < p_ds) & mb
            ds[:, 0] = False                                                       # BOS is never substituted / deleted
            dele = ds & (torch.rand_like(p_ds) < lam_del / (lam_sub + lam_del + 1e-12))
            sub = ds & ~dele
            ins_tok = torch.distributions.Categorical(logits=ins_logp).sample()
            sub_tok = torch.distributions.Categorical(logits=sub_logp).sample()
            xb = torch.where(sub, sub_tok, xb)
            # rebuild (vectorized apply_ins_del_operations): kept tokens shift by the insertions/deletions before them,
            # an insertion at position i lands right after token i (after BOS = at the front); capped at Lmax
            keep = mb & ~dele; keep[:, 0] = False                                  # BOS is not part of the output
            n_ins_before = ins.long().cumsum(1) - ins.long()                       # insertions at positions < i
            n_keep_before = keep.long().cumsum(1) - keep.long()                    # kept tokens at positions < i
            pos_keep = n_keep_before + n_ins_before                                 # output index of a kept token
            pos_ins = n_keep_before + keep.long() + n_ins_before                    # output index of the insertion after i
            new = torch.full((B, Lmax), tk.PAD, dtype=torch.long, device=dev)
            bi = torch.arange(B, device=dev).unsqueeze(1).expand_as(xb)
            ok_k = keep & (pos_keep < Lmax); ok_i = ins & (pos_ins < Lmax)
            new[bi[ok_k], pos_keep[ok_k]] = xb[ok_k]
            new[bi[ok_i], pos_ins[ok_i]] = ins_tok[ok_i]
            n_new = (new != tk.PAD).sum(1)
            x, n = new, n_new
            t += h
        return tk.decode(x, self.max_len)
