'''
Layout + text: LayoutFlowVarLen (per-element clocks) with a FlexMDM text factor (src/text_factor.py) on every
text element. The framework's instance: coordinate = (type: degenerate, box: Gaussian path, text: FlexMDM
insertion + unmasking), all on the element's clock; the text is born empty when the element is inserted and
clean when the element's box is.

Coupling per step (training and sampling alike): text -> layout through the pooled embeddings of the current
text tokens added to the element token (`elem_extra`); layout -> text through the soft prompt built from the
element's hidden state and the layout's global token. Losses: the layout losses + FlexMDM's unmasking and
insertion losses on the text elements.
'''
import math
import torch
import torch.nn as nn

from src.models.layout_flow_varlen import LayoutFlowVarLen


class PromptEncoder(nn.Module):
    '''
    The text factor's own view of the layout: a small transformer over the current element states (box at its noise
    level, class, clock) plus a global token. Trained by the text loss only, so the shared layout backbone is never
    pulled by the text objective. Returns per-element states (B,S,d) and the global state (B,d).
    '''
    def __init__(self, num_cat, d=512, nhead=8, layers=2):
        super().__init__()
        self.box = nn.Linear(4, d); self.cls = nn.Embedding(num_cat + 1, d); self.clock = nn.Linear(1, d)
        self.glob = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, 0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers); self.norm = nn.LayerNorm(d)

    def forward(self, x, y, visible, clocks):
        B, S = y.shape
        tok = self.box(x) + self.cls(y.clamp(min=0)) + self.clock(clocks.unsqueeze(-1).float())
        tok = torch.cat([self.glob.expand(B, -1, -1), tok], 1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), ~visible], 1)
        h = self.norm(self.enc(tok, src_key_padding_mask=pad))
        return h[:, 1:], h[:, 0]


class LayoutFlowText(LayoutFlowVarLen):
    def __init__(self, *args, text_factor=None, text_id=2, text_loss_weight=1.0, text_chunk=512, init_layout_ckpt=None, init_ckpt=None,
                 text_mode='joint', text_grad_to_layout=False, prompt_encoder=True, text_lr=1e-4, text_lag=0.0,
                 text_sample_chunk=96, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_lr = text_lr                        # learning rate of the LM's LoRA / insertion head (the rest uses the layout lr)
        # text_lag a: the text clock trails the element's box clock, u_text = clamp((u - a) / (1 - a), 0, 1), so tokens are
        # committed only once the box has settled (our FlexMDM port inserts and unmasks but never substitutes, so an early
        # commitment under a noisy layout is permanent). Both clocks still reach 1 together and the text still feeds back
        # into the layout. a = 0 is the fully concurrent model.
        self.text_lag = float(text_lag)
        self.text_sample_chunk = text_sample_chunk    # text sequences per LM forward while sampling (the vocab-sized
                                                      # logits are ~9 MB per sequence, so a whole batch does not fit)
        # prompt_encoder: the text factor reads the layout through its own encoder (PromptEncoder) instead of the
        # shared backbone's hidden states
        d = text_factor.pool[-1].out_features if text_factor is not None else 512
        self.prompt_enc = PromptEncoder(kwargs.get('num_cat', 6), d) if prompt_encoder else None
        # the text loss is ~60x the layout losses; letting its gradient reach the shared layout backbone (through the
        # soft prompt) slowly degrades the boxes (v3: flow loss 0.27 -> 0.40 over 50 epochs). Off: the prompt sees a
        # detached layout state; the layout stays text-aware through the pooled feature, trained by the layout losses.
        self.text_grad_to_layout = text_grad_to_layout
        self.text_factor = text_factor                # src.text_factor.TextFactor (hydra-instantiated)
        self.text_id, self.text_loss_weight, self.text_chunk = text_id, text_loss_weight, text_chunk
        # 'joint': text and boxes co-evolve on the element clocks, coupled both ways (the model);
        # 'caption': the layout-then-caption baseline -- the text factor is conditioned on the *clean* layout only, no
        #            text -> layout feature, and the text is sampled after the layout is finished (same LM, same LoRA)
        assert text_mode in ('joint', 'caption')
        self.text_mode = text_mode
        self.save_hyperparameters(ignore=['backbone_model', 'sampler', 'text_factor'])
        if init_ckpt:                                 # continue from a previous layout + text run (all trainable tensors)
            sd = torch.load(init_ckpt, map_location='cpu', weights_only=False)['state_dict']
            own = self.state_dict()
            sd = {k: v for k, v in sd.items() if k not in own or own[k].shape == v.shape}       # skip re-shaped heads
            res = self.load_state_dict(sd, strict=False)
            print(f'[LayoutFlowText] initialised from {init_ckpt}: {len(sd)} tensors, missing {len(res.missing_keys)} (frozen backbone), unexpected {len(res.unexpected_keys)}')
        if init_layout_ckpt:                          # start from a layout-only run of the same backbone
            sd = torch.load(init_layout_ckpt, map_location='cpu', weights_only=False)['state_dict']
            sd = {k: v for k, v in sd.items() if k.startswith('model.')}
            res = self.load_state_dict(sd, strict=False)
            print(f'[LayoutFlowText] layout backbone initialised from {init_layout_ckpt}: {len(sd)} tensors, unexpected {len(res.unexpected_keys)}')

    def configure_optimizers(self):
        # base.configure_optimizers optimises self.model (the layout backbone) only: the text factor's LoRA, insertion
        # head, prompt/pool heads and the prompt encoder must be added, the LM parts at a lower learning rate
        text_params = [p for n, p in self.named_parameters() if p.requires_grad and n.startswith('text_factor.model.')]
        other = [p for n, p in self.named_parameters() if p.requires_grad and not n.startswith('text_factor.model.')]
        lr = self.optimizer_partial.keywords.get('lr', 5e-4)
        optimizer = self.optimizer_partial(params=[{'params': other}, {'params': text_params, 'lr': self.text_lr if self.text_lr else lr}], betas=(0.9, 0.98))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        freq = self._trainer.check_val_every_n_epoch if getattr(self, '_trainer', None) is not None else 1
        return [optimizer], [{'scheduler': scheduler, 'monitor': 'val_loss', 'frequency': freq}]

    def state_dict(self, *args, **kwargs):
        # checkpoints carry only the trainable tensors (the frozen 7B backbone is re-loaded from the Hub)
        sd = super().state_dict(*args, **kwargs)
        frozen = {n for n, p in self.named_parameters() if not p.requires_grad}
        return {k: v for k, v in sd.items() if k not in frozen}

    def load_state_dict(self, sd, strict=True):
        return super().load_state_dict(sd, strict=False)

    def parameters(self, recurse=True):
        return (p for p in super().parameters(recurse) if p.requires_grad)

    def _text_clock(self, clocks):
        return clocks if self.text_lag <= 0 else ((clocks - self.text_lag) / (1 - self.text_lag)).clamp(0, 1)

    def _cond(self, h, h_glob, b, i):
        c = torch.cat([h[b, i], h_glob[b]], -1)
        # states from the prompt encoder are its own (trained by the text loss); states from the shared backbone are
        # detached unless text_grad_to_layout
        return c if (self.prompt_enc is not None or self.text_grad_to_layout) else c.detach()

    def _prompt_states(self, x, y, visible, clocks, h, h_glob):
        '''per-element / global states the prompt is built from: the prompt encoder's, or the backbone's (detached)'''
        if self.prompt_enc is not None:
            return self.prompt_enc(x, y, visible, clocks)
        return h, h_glob

    # ---------------- training ----------------
    def _text_pre(self, batch, visible, y, clocks, x=None):
        '''sample the FlexMDM state of every visible text element at its clock; return the layout-side feature'''
        tf = self.text_factor
        sel = visible & (y == self.text_id)
        self._prompt_in = (x, y, visible, clocks)
        if not sel.any():
            return None, None
        b_idx, i_idx = torch.nonzero(sel, as_tuple=True)
        strings = [batch['text'][b][i] for b, i in zip(b_idx.tolist(), i_idx.tolist())]
        x1, attn = tf.encode(strings, self.device)
        if self.text_mode == 'caption':               # own time, independent of the layout; no feature into the layout
            t = torch.rand(len(b_idx), device=self.device).clamp(1e-4, 1 - 1e-4)
            xt, xt_attn, st, masked, gaps, ins_mask = tf.noisy(x1, attn, t)
            return None, dict(b=b_idx, i=i_idx, x1=x1, xt=xt, attn=xt_attn, st=st, masked=masked, gaps=gaps, ins_mask=ins_mask, t=t, batch=batch)
        t = self._text_clock(clocks[b_idx, i_idx]).clamp(1e-4, 1 - 1e-4)
        xt, xt_attn, st, masked, gaps, ins_mask = tf.noisy(x1, attn, t)
        feat = torch.zeros(*visible.shape, tf.pool[-1].out_features, device=self.device)
        feat[b_idx, i_idx] = tf.pooled(xt, xt_attn).to(feat.dtype)
        return feat, dict(b=b_idx, i=i_idx, x1=x1, xt=xt, attn=xt_attn, st=st, masked=masked, gaps=gaps, ins_mask=ins_mask, t=t)

    def _text_post(self, state, h, h_glob):
        if state is None:
            return None
        tf = self.text_factor
        if self.text_mode == 'caption':               # condition on the clean layout
            h, h_glob = self._clean_layout_states(state['batch'])
        else:
            h, h_glob = self._prompt_states(*self._prompt_in, h, h_glob)
        cond = self._cond(h, h_glob, state['b'], state['i'])
        unmask_loss = ins_loss = 0.0
        n = state['xt'].shape[0]
        for s in range(0, n, self.text_chunk):
            sl = slice(s, s + self.text_chunk)
            logits, log_length = tf(state['xt'][sl], state['attn'][sl], state['t'][sl], cond[sl])
            u, i = tf.losses(state['x1'][sl], state['st'][sl], state['masked'][sl], state['gaps'][sl], state['ins_mask'][sl], state['t'][sl], logits, log_length)
            w = min(self.text_chunk, n - s) / n
            unmask_loss = unmask_loss + w * u; ins_loss = ins_loss + w * i
        self.log('text_unmask_loss', unmask_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log('text_ins_loss', ins_loss, on_step=True, on_epoch=True, sync_dist=True)
        return self.text_loss_weight * (unmask_loss + ins_loss)

    def _clean_layout_states(self, batch, x1=None, y1=None, active=None):
        '''layout backbone on the finished layout (t = 1, every element visible and clean)'''
        if x1 is None:
            active = batch['mask'].squeeze(-1); y1 = batch['type'].long(); x1 = active.unsqueeze(-1) * self.sampler.preprocess(batch['bbox'])
        B = x1.shape[0]
        if self.prompt_enc is not None:
            return self.prompt_enc(x1, y1, active, active.float())
        t = torch.ones(B, device=self.device); t_elem = active.float() if self.oneflow else None
        _, _, h, _, extra = self(x1, y1, active, t, batch.get('ctx'), None, t_elem, None, batch.get('ret'), None, batch.get('sal'))
        return h, extra['h_glob']

    # ---------------- sampling ----------------
    def _text_infer_init(self, batch):
        if self.text_mode == 'caption':
            return {'batch': batch}
        B, S = batch['type'].shape
        tf = self.text_factor
        xt, attn = tf.empty_state(B * S, self.device)
        return dict(xt=xt.view(B, S, -1), attn=attn.view(B, S, -1), alive=torch.zeros(B, S, dtype=torch.bool, device=self.device))

    def _text_infer_pre(self, state, visible, y, clocks):
        if self.text_mode == 'caption':
            return None
        tf = self.text_factor
        sel = visible & (y == self.text_id)
        # elements that just became visible text elements start from the empty state
        new = sel & ~state['alive']
        if new.any():
            n = int(new.sum()); xt0, a0 = tf.empty_state(n, self.device)
            state['xt'][new] = xt0; state['attn'][new] = a0
        state['alive'] = sel
        if not sel.any():
            return None
        feat = torch.zeros(*visible.shape, tf.pool[-1].out_features, device=self.device)
        feat[sel] = tf.pooled(state['xt'][sel], state['attn'][sel]).to(feat.dtype)
        return feat

    def _text_infer_step(self, state, h, h_glob, visible, y, clocks, ds, last, x=None):
        if self.text_mode == 'caption':
            return state
        tf = self.text_factor
        sel = state['alive']
        if not sel.any():
            return state
        b_idx, i_idx = torch.nonzero(sel, as_tuple=True)
        t_all = self._text_clock(clocks)
        if self.text_lag > 0 and not (t_all[sel] > 0).any():
            return state                                    # the text has not started yet on any live element
        h, h_glob = self._prompt_states(x, y, visible, clocks, h, h_glob)
        cond = self._cond(h, h_glob, b_idx, i_idx)
        t = t_all[b_idx, i_idx].clamp(0, 1 - 1e-4)
        d = ds[b_idx, i_idx] / (1 - self.text_lag) if self.text_lag > 0 else ds[b_idx, i_idx]
        d = d * (t_all[b_idx, i_idx] > 0).float()           # no text steps before the lag
        xt_all, attn_all = state['xt'][sel], state['attn'][sel]
        outs = []
        for s0 in range(0, xt_all.shape[0], self.text_sample_chunk):
            sl = slice(s0, s0 + self.text_sample_chunk)
            outs.append(tf.step(xt_all[sl], attn_all[sl], t[sl], d[sl], cond[sl], last=last))
        state['xt'][sel] = torch.cat([o[0] for o in outs]); state['attn'][sel] = torch.cat([o[1] for o in outs])
        return state

    def _text_infer_finish(self, state, exists, order, x=None, y=None, ctx=None, ret=None, sal=None):
        tf = self.text_factor
        B, S = exists.shape
        if self.text_mode == 'caption':               # caption the finished layout: N text steps with a fixed condition
            batch = dict(state['batch']); batch.update(ctx=ctx, ret=ret, sal=sal)
            h, h_glob = self._clean_layout_states(batch, x, y, exists)
            sel = exists & (y == self.text_id)
            xt_all, attn_all = tf.empty_state(B * S, self.device); xt_all, attn_all = xt_all.view(B, S, -1), attn_all.view(B, S, -1)
            if sel.any():
                b_idx, i_idx = torch.nonzero(sel, as_tuple=True)
                cond = self._cond(h, h_glob, b_idx, i_idx)
                N = self.inference_steps
                parts = []
                for s0 in range(0, len(b_idx), self.text_sample_chunk):
                    sl = slice(s0, s0 + self.text_sample_chunk); m = min(self.text_sample_chunk, len(b_idx) - s0)
                    xt, attn = tf.empty_state(m, self.device); dt = torch.full((m,), 1.0 / N, device=self.device)
                    for k in range(N):
                        xt, attn = tf.step(xt, attn, torch.full((m,), k / N, device=self.device), dt, cond[sl], last=(k == N - 1))
                    parts.append((xt, attn))
                xt_all[sel] = torch.cat([p[0] for p in parts]); attn_all[sel] = torch.cat([p[1] for p in parts])
            state = dict(xt=xt_all, attn=attn_all, alive=sel)
        xt = state['xt'].gather(1, order.unsqueeze(-1).expand_as(state['xt']))
        attn = state['attn'].gather(1, order.unsqueeze(-1).expand_as(state['attn']))
        alive = state['alive'].gather(1, order)
        texts = [[''] * S for _ in range(B)]
        if alive.any():
            b_idx, i_idx = torch.nonzero(alive, as_tuple=True)
            for b, i, s in zip(b_idx.tolist(), i_idx.tolist(), tf.decode(xt[alive], attn[alive])):
                texts[b][i] = s
        return texts
