from dataclasses import dataclass
import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass
class VariableLengthMaskingResult:
    xt: Tensor          # (B, L+1, geom_dim) repacked, alive-first
    yt: Tensor          # (B, L+1) repacked
    alive: Tensor       # (B, L+1) repacked bool -- "exists right now"
    is_masked: Tensor   # (B, L+1) repacked bool
    x1: Tensor          # (B, L+1, geom_dim) repacked ground truth
    y1: Tensor          # (B, L+1) repacked ground truth
    st: Tensor          # (B, L+1) permutation used to repack (values = original-order indices, in rank order)
    y1_len: Tensor      # (B,) total elements incl. BOS -- order-invariant, same value pre/post repack
    yt_len: Tensor      # (B,) currently-alive count incl. BOS
    t: Tensor


class VariableLengthMultimodalInterpolant:
    '''
    Genuinely variable-length version of MultimodalMaskingInterpolant: instead of
    every element existing from t=0 on a fixed-length canvas (unmasking only),
    elements are BORN (inserted) over time too, via a coupled
    masking_time >= deletion_time draw per original-order slot and a learned
    Poisson insertion-rate head. Ported from the pre-LayoutFlow-minimal
    multimodal_interpolant.py's core mechanism (BOS anchor that's always alive
    and never masked, argsort-based active-prefix repacking each step, and the
    jump_kernel_elbo rate-matching loss comparing the model's predicted
    insertion rate against the actual count of not-yet-revealed elements
    queued after each alive position). Simplified relative to that reference:
    one geometry head (not a per-category-conditional [B,L,V,D] head) and no
    CFG-dropout / cat_cond / size_cond conditioning branches -- unconditional
    generation only, matching the rest of this codebase's minimal style.

    The original dataset order of a sample's real elements is treated as its
    fixed (if arbitrary) canonical arrival order: element k in that order is
    the k-th element to ever become alive. deletion_time = 1 - u1 marks when a
    slot is born; masking_time = 1 - u1*u2 (always >= deletion_time) marks
    when it's revealed. Sorting alive-ness by descending value each step packs
    currently-alive elements to the front, in that same fixed relative order.
    '''

    def __init__(self, geom_dim: int, num_cat: int):
        self.geom_dim = geom_dim
        self.num_cat = num_cat
        self.mask_token = num_cat
        self.pad_token = num_cat + 1
        self.bos_token = num_cat + 2
        self.vocab_size = num_cat + 3

    def alpha(self, t):
        return t

    def dalpha(self, t):
        return torch.ones_like(t)

    def sample_time(self, batch_size, device):
        eps = 1e-5
        return torch.rand(batch_size, device=device) * (1 - eps)

    def prepend_bos(self, x1, y1, active):
        x1 = torch.cat([torch.zeros_like(x1[:, :1]), x1], dim=1)
        y1 = torch.cat([torch.full_like(y1[:, :1], self.bos_token), y1], dim=1)
        active = torch.cat([torch.ones_like(active[:, :1]), active], dim=1)
        return x1, y1, active

    def get_masking_and_deletion_time(self, shape, device):
        u1 = torch.rand(shape, device=device)
        deletion_time = 1 - u1
        u2 = torch.rand(shape, device=device)
        masking_time = 1 - u1 * u2   # always >= deletion_time
        return masking_time, deletion_time

    def sample_interpolant(self, t: Tensor, x1: Tensor, y1: Tensor, active: Tensor) -> VariableLengthMaskingResult:
        B, L = active.shape
        masking_time, deletion_time = self.get_masking_and_deletion_time((B, L), x1.device)

        alive = active & (t.unsqueeze(-1) >= deletion_time)
        alive[:, 0] = True   # BOS always alive
        is_masked = alive & (t.unsqueeze(-1) < masking_time)
        is_masked[:, 0] = False   # BOS never masked
        revealed = alive & ~is_masked

        alpha_t = self.alpha(t).view(-1, 1, 1)
        noise = torch.randn_like(x1)
        xt = alpha_t * x1 + (1 - alpha_t) * noise
        xt = torch.where(revealed.unsqueeze(-1), xt, torch.zeros_like(xt))

        yt = torch.where(is_masked, torch.full_like(y1, self.mask_token), y1)
        yt = torch.where(alive, yt, torch.full_like(y1, self.pad_token))

        y1_len = active.sum(dim=1)
        yt_len = alive.sum(dim=1)

        st = alive.long().argsort(dim=1, descending=True, stable=True)

        def gather1(v):
            return torch.gather(v, 1, st)

        def gather3(v):
            return torch.gather(v, 1, st.unsqueeze(-1).expand(-1, -1, v.shape[-1]))

        return VariableLengthMaskingResult(
            xt=gather3(xt), yt=gather1(yt), alive=gather1(alive), is_masked=gather1(is_masked),
            x1=gather3(x1), y1=gather1(y1), st=st, y1_len=y1_len, yt_len=yt_len, t=t,
        )

    def gaps_and_mask(self, st: Tensor, y1_len: Tensor, yt_len: Tensor):
        '''For each currently-alive rank k, how many not-yet-alive elements are
        queued to appear before the next alive rank -- the target the
        insertion-rate head is trained to predict via jump_kernel_elbo.'''
        gaps = st.clone()
        pad_back = gaps.new_zeros((gaps.shape[0], 1))
        gaps = torch.cat([gaps, pad_back], dim=1)
        gaps.scatter_(1, yt_len.unsqueeze(1), y1_len.unsqueeze(1))
        gaps = gaps[:, 1:] - gaps[:, :-1] - 1
        gaps = torch.clamp(gaps, min=0)
        idx = torch.arange(gaps.size(1), device=gaps.device).unsqueeze(0)
        mask = idx < yt_len.unsqueeze(1)
        gaps = torch.where(mask, gaps, torch.zeros_like(gaps))
        return gaps, mask

    def jump_kernel_elbo(self, observed: Tensor, predicted_rate: Tensor, eps=1e-6):
        obs = observed.clamp(min=eps)
        pred = predicted_rate.clamp(min=eps)
        return pred - obs + obs * (torch.log(obs) - torch.log(pred))

    def compute_loss(self, model, batch) -> dict:
        x1_raw = batch['x']; y1_raw = batch['y']; active_raw = batch['mask']
        x1, y1, active = self.prepend_bos(x1_raw, y1_raw, active_raw)
        t = self.sample_time(x1.shape[0], x1.device)
        sample = self.sample_interpolant(t, x1, y1, active)

        geom_pred, cat_logits, insertion_rate = model(sample.xt, sample.yt, sample.is_masked, sample.alive, t)

        revealed = sample.alive & ~sample.is_masked
        geom_err = (sample.x1 - geom_pred).pow(2).sum(dim=-1)[revealed]
        geom_loss = geom_err.mean() / self.geom_dim if geom_err.numel() > 0 else geom_pred.sum() * 0

        masked = sample.alive & sample.is_masked
        if masked.any():
            cat_loss = F.cross_entropy(cat_logits[masked], sample.y1[masked].long(), reduction='mean')
        else:
            cat_loss = cat_logits.sum() * 0

        gaps, gaps_mask = self.gaps_and_mask(sample.st, sample.y1_len, sample.yt_len)
        if gaps_mask.any():
            insertion_loss = self.jump_kernel_elbo(gaps[gaps_mask].float(), insertion_rate[gaps_mask]).mean()
        else:
            insertion_loss = insertion_rate.sum() * 0

        return {'geom_loss': geom_loss, 'cat_loss': cat_loss, 'insertion_loss': insertion_loss}

    def _perform_insertions(self, xt, yt, alive, ins_counts, max_length):
        B = xt.shape[0]
        device = xt.device
        new_xt = torch.zeros_like(xt)
        new_yt = torch.full_like(yt, self.pad_token)
        new_alive = torch.zeros_like(alive)
        for b in range(B):
            xs, ys = [], []
            for k in range(max_length):
                if not bool(alive[b, k]):
                    break
                xs.append(xt[b, k])
                ys.append(int(yt[b, k].item()))
                for _ in range(int(ins_counts[b, k].item())):
                    if len(xs) >= max_length:
                        break
                    xs.append(torch.zeros_like(xt[b, 0]))
                    ys.append(self.mask_token)
            n = min(len(xs), max_length)
            if n > 0:
                new_xt[b, :n] = torch.stack(xs[:n])
                new_yt[b, :n] = torch.tensor(ys[:n], device=device, dtype=yt.dtype)
                new_alive[b, :n] = True
            new_xt[b, 0] = 0.
            new_yt[b, 0] = self.bos_token
            new_alive[b, 0] = True
        return new_xt, new_yt, new_alive

    @torch.no_grad()
    def sampling(self, model, num_steps: int, batch_size: int, max_length: int, device):
        L = max_length + 1   # +1 for BOS
        xt = torch.zeros((batch_size, L, self.geom_dim), device=device)
        yt = torch.full((batch_size, L), self.pad_token, dtype=torch.long, device=device)
        yt[:, 0] = self.bos_token
        alive = torch.zeros((batch_size, L), dtype=torch.bool, device=device)
        alive[:, 0] = True
        ts = torch.linspace(0, 1, num_steps + 1, device=device)[:-1]
        dt = 1.0 / num_steps

        for i, t_scalar in enumerate(ts):
            t = t_scalar.expand(batch_size)
            is_last_step = i == num_steps - 1
            is_masked = alive & (yt == self.mask_token)
            geom_pred, cat_logits, insertion_rate = model(xt, yt, is_masked, alive, t)

            revealed = alive & ~is_masked
            drift = (geom_pred - xt) / (1 - self.alpha(t).view(-1, 1, 1)).clamp(min=1e-5)
            xt = torch.where(revealed.unsqueeze(-1), xt + drift * dt, xt)

            masked = alive & is_masked
            if is_last_step:
                reveal = masked
            else:
                rate = (self.dalpha(t) / (1 - self.alpha(t)).clamp(min=1e-5)).view(-1, 1)
                num_events = torch.distributions.Poisson(rate * dt).sample()
                reveal = masked & (num_events > 0)

            cat_sample = torch.distributions.Categorical(logits=cat_logits).sample()
            yt = torch.where(reveal, cat_sample, yt)
            alpha_t = self.alpha(t).view(-1, 1, 1)
            new_geom = alpha_t * geom_pred + (1 - alpha_t) * torch.randn_like(xt)
            xt = torch.where(reveal.unsqueeze(-1), new_geom, xt)
            is_masked = is_masked & ~reveal

            if not is_last_step:
                ins_coeff = (self.dalpha(t) / (1 - self.alpha(t)).clamp(min=1e-5)).view(-1, 1)
                ins_rate = ins_coeff * insertion_rate
                ins_counts = torch.distributions.Poisson(ins_rate * dt).sample().long()
                ins_counts = torch.where(alive, ins_counts, torch.zeros_like(ins_counts))
                xt, yt, alive = self._perform_insertions(xt, yt, alive, ins_counts, L)

        return xt, yt, alive
