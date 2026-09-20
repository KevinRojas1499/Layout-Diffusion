'''
Text factor of a layout element: FlexMDM (insertion + unmasking masked diffusion, Kim et al. 2025) on the released
Dream-Coder-7B checkpoint (yuyuanchen0/flexmdm), LoRA-adapted and conditioned on the layout through a soft prompt.

Each text element is its own short sequence [K prefix positions | answer tokens]. The prefix positions are FlexMDM
"prompt" tokens (never deleted or masked; insertions happen after the last one) whose embeddings are produced from
the layout state (the element's hidden state and the global token of the layout backbone). The text clock is the
element's clock: the answer is empty at 0 and clean at 1, with FlexMDM's power schedules (insertion exponent a,
unmasking exponent a*b) run on that clock. Sampling is FlexMDM's Algorithm 1 (confidence top-k unmasking + Poisson
insertion) ported to a per-sequence step size, so that a text advances one step per layout step on its own clock.

Needs the FlexMDM package (github SeunggeunKimkr/genuine-any-order, FlexMDM/) and transformers 4.46.x.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextFactor(nn.Module):
    def __init__(self, ckpt='yuyuanchen0/flexmdm', d_layout=512, n_prefix=4, max_tokens=24, lora_r=16, lora_alpha=32,
                 lora_dropout=0.05, lora_targets=('q_proj', 'k_proj', 'v_proj', 'o_proj'), train_extras=True,
                 insertion_exponent=1.7, unmasking_exponent=2.89, temperature=0.1, insertion_temperature=1.0,
                 gradient_checkpointing=True, dtype='bfloat16'):
        super().__init__()
        from huggingface_hub import snapshot_download
        from flexmdm.utils import load_model_and_tokenizer
        from flexmdm.trainer import FlexMDMProcess
        from peft import LoraConfig, inject_adapter_in_model
        self.K, self.max_tokens = n_prefix, max_tokens
        self.L = n_prefix + max_tokens
        self.model, self.tok = load_model_and_tokenizer(checkpoint_dir=snapshot_download(ckpt), max_length=self.L, torch_dtype_name=dtype)
        self.mask_id, self.pad_id = self.tok.mask_token_id, self.tok.pad_token_id
        H = self.model.hidden_size
        for p in self.model.parameters():
            p.requires_grad_(False)
        inject_adapter_in_model(LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, target_modules=list(lora_targets)), self.model.backbone)
        if train_extras:                                  # FlexMDM's insertion head (13M); the time MLP / AdaLN modulators (820M) stay frozen
            for p in self.model.insertion_head.parameters():
                p.requires_grad_(True)
        if gradient_checkpointing and hasattr(self.model.backbone, 'gradient_checkpointing_enable'):
            self.model.backbone.gradient_checkpointing_enable()
        # layout -> text: K soft prompt tokens from [h_elem, h_glob]; text -> layout: pooled token embeddings
        self.prefix = nn.Sequential(nn.Linear(2 * d_layout, H), nn.SiLU(), nn.Linear(H, n_prefix * H))
        self.pool = nn.Sequential(nn.Linear(H, d_layout), nn.SiLU(), nn.Linear(d_layout, d_layout))
        nn.init.zeros_(self.pool[-1].weight); nn.init.zeros_(self.pool[-1].bias)     # the layout side starts text-blind
        self.process = FlexMDMProcess(vocab_size=self.model.backbone.config.vocab_size, mask_id=self.mask_id, pad_id=self.pad_id,
                                      max_len=self.L, insertion_schedule='power', unmasking_schedule='power',
                                      insertion_exponent=insertion_exponent, unmasking_exponent=unmasking_exponent)
        self.ins_exp, self.unm_exp = insertion_exponent, unmasking_exponent
        self.temperature, self.ins_temperature = temperature, insertion_temperature
        self.n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ---------------- tokens ----------------
    def encode(self, strings, device):
        '''list of strings -> x1 (N, L) with the K prompt placeholders in front, and its attention mask'''
        ids = self.tok(list(strings), add_special_tokens=False, truncation=True, max_length=self.max_tokens)['input_ids']
        x1 = torch.full((len(ids), self.L), self.pad_id, dtype=torch.long, device=device)
        attn = torch.zeros(len(ids), self.L, dtype=torch.bool, device=device)
        x1[:, :self.K] = self.pad_id; attn[:, :self.K] = True
        for n, seq in enumerate(ids):
            x1[n, self.K:self.K + len(seq)] = torch.tensor(seq, device=device); attn[n, self.K:self.K + len(seq)] = True
        return x1, attn

    def decode(self, xt, attn):
        out = []
        for n in range(xt.shape[0]):
            seq = xt[n, self.K:][attn[n, self.K:]]
            out.append(self.tok.decode(seq[(seq != self.mask_id) & (seq != self.pad_id)]))
        return out

    def empty_state(self, n, device):
        '''prefix only: the text at clock 0'''
        xt = torch.full((n, self.L), self.pad_id, dtype=torch.long, device=device)
        attn = torch.zeros(n, self.L, dtype=torch.bool, device=device); attn[:, :self.K] = True
        return xt, attn

    # ---------------- network ----------------
    def prompt_mask(self, attn):
        m = torch.zeros_like(attn); m[:, :self.K] = True
        return m

    def forward(self, xt, attn, t, cond):
        '''xt (N,L), attn (N,L) bool, t (N,) in [0,1], cond (N, 2 d_layout) -> logits (N,L,V), log_length (N,L)'''
        m = self.model
        emb = m.backbone.model.embed_tokens(xt)
        pre = self.prefix(cond.float()).to(emb.dtype).view(xt.shape[0], self.K, -1)
        emb = torch.cat([pre, emb[:, self.K:]], 1)
        temb = m.timestep_embedding(t.to(xt.device).float()).to(emb.dtype)
        m._temb = m.time_mlp(temb)
        out = m.backbone.model(inputs_embeds=emb, attention_mask=attn.bool()[:, None, None, :], return_dict=True)
        last = out.last_hidden_state if hasattr(out, 'last_hidden_state') else out[0]
        logits, log_length = m.backbone.lm_head(last), m.insertion_head(last)
        m._temb = None
        return logits, log_length

    def pooled(self, xt, attn):
        '''text -> layout feature (N, d_layout) from the frozen token embeddings of the current answer tokens'''
        with torch.no_grad():
            emb = self.model.backbone.model.embed_tokens(xt).float()
        w = attn.float().clone(); w[:, :self.K] = 0
        mean = (emb * w.unsqueeze(-1)).sum(1) / w.sum(1, keepdim=True).clamp(min=1)
        return self.pool(mean.to(self.pool[0].weight.dtype))

    # ---------------- training ----------------
    def noisy(self, x1, attn, t):
        '''FlexMDM's conditional path at time t: (xt, xt_attn, st, masked_indices, gaps, insertion_mask)'''
        xt, xt_attn, st, _, masked, gaps, ins_mask = self.process.flexmdm_process(x1, t.float(), self.prompt_mask(attn), attn)
        return xt, xt_attn, st, masked, gaps, ins_mask

    def losses(self, x1, st, masked, gaps, ins_mask, t, logits, log_length):
        '''FlexMDM's unmasking cross-entropy + insertion Bregman loss (their trainer.loss, on the given outputs)'''
        from flexmdm.trainer import poisson_loss
        w_ins, w_unm = self.process.elbo_weights(t.float())
        unmask_pred = torch.cat([logits[:, :1], logits[:, :-1]], 1)             # Dream-style shift: position i predicts i + 1
        target = self.process.gathered_unmasked(x1, st)
        n = max(int(masked.sum()), 1)
        unmask_loss = (w_unm[:, None].expand_as(masked)[masked] * F.cross_entropy(unmask_pred[masked].float(), target[masked], reduction='none')).sum() / n
        n_i = max(int(ins_mask.sum()), 1)
        ins_loss = (w_ins[:, None].expand_as(ins_mask)[ins_mask].float() * poisson_loss(gaps[ins_mask], log_length[ins_mask])).sum() / n_i
        return unmask_loss, ins_loss

    # ---------------- sampling (Algorithm 1 with a per-sequence dt) ----------------
    @torch.no_grad()
    def step(self, xt, attn, t, dt, cond, last=False):
        from flexmdm.schedules import schedule_hazard_rate
        from flexmdm.inference import _apply_after_token_insertions
        logits, log_length = self.forward(xt, attn, t, cond)
        unmask_logits = torch.cat([logits[:, :1], logits[:, :-1]], 1).float()
        masked = xt.eq(self.mask_id) & attn
        N, L = xt.shape
        if last:
            final = unmask_logits.argmax(-1)
            xt = torch.where(masked, final, xt)
            return xt, attn
        # unmask: sample clean tokens, reveal Poisson(masked_count * hazard * dt) of them by confidence
        if self.temperature > 0:
            u = torch.rand_like(unmask_logits).clamp(min=1e-20)
            g = -torch.log((-torch.log(u)).clamp(min=1e-20))                      # Gumbel(0, 1)
            sampled = (unmask_logits / self.temperature + g).argmax(-1)
        else:
            sampled = unmask_logits.argmax(-1)
        probs = F.softmax(unmask_logits, -1)
        conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1).masked_fill(~masked, -1.0)
        rate_u = schedule_hazard_rate(t.float(), schedule='power', exponent=self.unm_exp, field='unmasking hazard_rate')
        k = torch.poisson(masked.sum(1).float() * rate_u * dt.float()).long().clamp(max=L)
        rank = conf.argsort(1, descending=True).argsort(1)                     # rank of each position by confidence
        reveal = masked & (rank < k.unsqueeze(1))
        xt = torch.where(reveal, sampled, xt)
        # insert: Poisson(exp(g) * hazard * dt) new mask tokens after each valid position (after the last prompt token on)
        rate_i = schedule_hazard_rate(t.float(), schedule='power', exponent=self.ins_exp, field='insertion hazard_rate')
        length = attn.sum(1)
        pos = torch.arange(L, device=xt.device).unsqueeze(0)
        valid = attn & (pos >= self.K - 1)
        gval = log_length.clamp(max=15.0).float().masked_fill(~valid, float('-inf'))
        lam_total = gval.exp().sum(1, keepdim=True) * (rate_i * dt.float()).unsqueeze(1)
        placement = F.softmax(gval / self.ins_temperature, 1)
        expected = torch.where(lam_total > 0, lam_total * placement, torch.zeros_like(placement))
        ins = torch.poisson(expected.nan_to_num(0.0)).long()
        # cap at the free capacity, first-come
        cap = (L - length).clamp(min=0)
        cum = ins.cumsum(1)
        ins = torch.where(cum <= cap.unsqueeze(1), ins, (cap.unsqueeze(1) - (cum - ins)).clamp(min=0))
        if ins.any():
            xt, attn, _, _ = _apply_after_token_insertions(xt, attn, ins, pad_id=self.pad_id, mask_id=self.mask_id)
        return xt, attn
