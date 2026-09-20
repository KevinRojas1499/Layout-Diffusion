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
import torch

from src.models.layout_flow_varlen import LayoutFlowVarLen


class LayoutFlowText(LayoutFlowVarLen):
    def __init__(self, *args, text_factor=None, text_id=2, text_loss_weight=1.0, text_chunk=512, init_layout_ckpt=None, init_ckpt=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_factor = text_factor                # src.text_factor.TextFactor (hydra-instantiated)
        self.text_id, self.text_loss_weight, self.text_chunk = text_id, text_loss_weight, text_chunk
        self.save_hyperparameters(ignore=['backbone_model', 'sampler', 'text_factor'])
        if init_ckpt:                                 # continue from a previous layout + text run (all trainable tensors)
            sd = torch.load(init_ckpt, map_location='cpu', weights_only=False)['state_dict']
            res = self.load_state_dict(sd, strict=False)
            print(f'[LayoutFlowText] initialised from {init_ckpt}: {len(sd)} tensors, missing {len(res.missing_keys)} (frozen backbone), unexpected {len(res.unexpected_keys)}')
        if init_layout_ckpt:                          # start from a layout-only run of the same backbone
            sd = torch.load(init_layout_ckpt, map_location='cpu', weights_only=False)['state_dict']
            sd = {k: v for k, v in sd.items() if k.startswith('model.')}
            res = self.load_state_dict(sd, strict=False)
            print(f'[LayoutFlowText] layout backbone initialised from {init_layout_ckpt}: {len(sd)} tensors, unexpected {len(res.unexpected_keys)}')

    def state_dict(self, *args, **kwargs):
        # checkpoints carry only the trainable tensors (the frozen 7B backbone is re-loaded from the Hub)
        sd = super().state_dict(*args, **kwargs)
        frozen = {n for n, p in self.named_parameters() if not p.requires_grad}
        return {k: v for k, v in sd.items() if k not in frozen}

    def load_state_dict(self, sd, strict=True):
        return super().load_state_dict(sd, strict=False)

    def parameters(self, recurse=True):
        return (p for p in super().parameters(recurse) if p.requires_grad)

    def _cond(self, h, h_glob, b, i):
        return torch.cat([h[b, i], h_glob[b]], -1)

    # ---------------- training ----------------
    def _text_pre(self, batch, visible, y, clocks):
        '''sample the FlexMDM state of every visible text element at its clock; return the layout-side feature'''
        tf = self.text_factor
        sel = visible & (y == self.text_id)
        if not sel.any():
            return None, None
        b_idx, i_idx = torch.nonzero(sel, as_tuple=True)
        strings = [batch['text'][b][i] for b, i in zip(b_idx.tolist(), i_idx.tolist())]
        x1, attn = tf.encode(strings, self.device)
        t = clocks[b_idx, i_idx].clamp(1e-4, 1 - 1e-4)
        xt, xt_attn, st, masked, gaps, ins_mask = tf.noisy(x1, attn, t)
        feat = torch.zeros(*visible.shape, tf.pool[-1].out_features, device=self.device)
        feat[b_idx, i_idx] = tf.pooled(xt, xt_attn).to(feat.dtype)
        return feat, dict(b=b_idx, i=i_idx, x1=x1, xt=xt, attn=xt_attn, st=st, masked=masked, gaps=gaps, ins_mask=ins_mask, t=t)

    def _text_post(self, state, h, h_glob):
        if state is None:
            return None
        tf = self.text_factor
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

    # ---------------- sampling ----------------
    def _text_infer_init(self, batch):
        B, S = batch['type'].shape
        tf = self.text_factor
        xt, attn = tf.empty_state(B * S, self.device)
        return dict(xt=xt.view(B, S, -1), attn=attn.view(B, S, -1), alive=torch.zeros(B, S, dtype=torch.bool, device=self.device))

    def _text_infer_pre(self, state, visible, y, clocks):
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

    def _text_infer_step(self, state, h, h_glob, visible, y, clocks, ds, last):
        tf = self.text_factor
        sel = state['alive']
        if not sel.any():
            return state
        b_idx, i_idx = torch.nonzero(sel, as_tuple=True)
        cond = self._cond(h, h_glob, b_idx, i_idx)
        t = clocks[b_idx, i_idx].clamp(0, 1 - 1e-4)
        d = ds[b_idx, i_idx]
        xt, attn = tf.step(state['xt'][sel], state['attn'][sel], t, d, cond, last=last)
        state['xt'][sel] = xt; state['attn'][sel] = attn
        return state

    def _text_infer_finish(self, state, exists, order):
        tf = self.text_factor
        B, S = exists.shape
        xt = state['xt'].gather(1, order.unsqueeze(-1).expand_as(state['xt']))
        attn = state['attn'].gather(1, order.unsqueeze(-1).expand_as(state['attn']))
        alive = state['alive'].gather(1, order)
        texts = [[''] * S for _ in range(B)]
        if alive.any():
            b_idx, i_idx = torch.nonzero(alive, as_tuple=True)
            for b, i, s in zip(b_idx.tolist(), i_idx.tolist(), tf.decode(xt[alive], attn[alive])):
                texts[b][i] = s
        return texts
