from __future__ import annotations
import torch
from torch import Tensor
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import torch.nn.functional as F
from tqdm import tqdm


@dataclass
class BranchingFlowsPrediction:
    # This handles denoising
    clean_data: Tensor
    # This handles the unmasking
    label_logits: Tensor
    clean_data_unmasking: Tensor
    # This handles the insertions
    insertion_rate: Tensor

@dataclass
class SamplingTrajectoryResult:
    xt: Tensor # Shape [Batch, Length]
    yt: Tensor # Shape [Batch, Length]
    mask_t: Tensor # Shape [Batch, Length]
    t: Tensor # Shape [Batch]

@dataclass
class SamplingResult:
    xt: Tensor # Shape [Batch, Length]
    yt: Tensor # Shape [Batch, Length]
    mask_t: Tensor # Shape [Batch, Length]
    trajectory: List[SamplingTrajectoryResult]

    def __getitem__(self, index: int) -> SamplingTrajectoryResult:
        trajectory_slice = []
        for step in self.trajectory:
             trajectory_slice.append(SamplingTrajectoryResult(
                xt=step.xt[index],
                yt=step.yt[index],
                mask_t=step.mask_t[index],
                t=step.t[index]
             ))

        return SamplingResult(
            xt=self.xt[index],
            yt=self.yt[index],
            mask_t=self.mask_t[index],
            trajectory=trajectory_slice
        )

    def __len__(self) -> int:
        return self.xt.shape[0]
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


# ============================================================
# Same tree + sampler utilities as before (kept mostly intact)
# ============================================================

@dataclass
class Node:
    left: Optional["Node"] = None
    right: Optional["Node"] = None

    leaf_idx: Optional[int] = None
    leaf_deleted: bool = False

    anchor: Optional[torch.Tensor] = None
    split_time: Optional[float] = None
    del_time: Optional[float] = None

    n_desc: Optional[int] = None

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


def _assign_leaf_indices_inorder(node: Node, start: int = 0) -> int:
    if node.is_leaf():
        node.leaf_idx = start
        return start + 1
    nxt = _assign_leaf_indices_inorder(node.left, start)
    nxt = _assign_leaf_indices_inorder(node.right, nxt)
    return nxt


def _compute_desc_counts(node: Node) -> int:
    if node.is_leaf():
        node.n_desc = 1
    else:
        node.n_desc = _compute_desc_counts(node.left) + _compute_desc_counts(node.right)
    return node.n_desc


def sample_random_full_binary_tree(num_leaves: int) -> Node:
    if num_leaves < 1:
        raise ValueError("num_leaves must be >= 1")
    nodes: List[Node] = [Node() for _ in range(num_leaves)]
    while len(nodes) > 1:
        i = int(torch.randint(0, len(nodes) - 1, (1,)).item())
        parent = Node(left=nodes[i], right=nodes[i + 1])
        nodes = nodes[:i] + [parent] + nodes[i + 2 :]
    root = nodes[0]
    _assign_leaf_indices_inorder(root, 0)
    _compute_desc_counts(root)
    return root


def assign_leaf_deleted_flags(root: Node, n_deleted: int) -> None:
    leaves: List[Node] = []
    stack = [root]
    while stack:
        u = stack.pop()
        if u.is_leaf():
            leaves.append(u)
        else:
            stack.append(u.right)
            stack.append(u.left)

    if n_deleted > len(leaves):
        raise ValueError("n_deleted exceeds number of leaves")
    if n_deleted == 0:
        return

    idx = torch.randperm(len(leaves))[:n_deleted].tolist()
    chosen = set(idx)
    for j, leaf in enumerate(leaves):
        leaf.leaf_deleted = (j in chosen)


def assign_anchors_bottom_up(root: Node, x1_plus: torch.Tensor, merge: str = "weighted_by_desc") -> None:
    def post(u: Node) -> torch.Tensor:
        if u.is_leaf():
            assert u.leaf_idx is not None
            u.anchor = x1_plus[u.leaf_idx].clone()
            return u.anchor
        aL = post(u.left)
        aR = post(u.right)
        if merge == "mean":
            u.anchor = 0.5 * (aL + aR)
        elif merge == "weighted_by_desc":
            wL, wR = float(u.left.n_desc), float(u.right.n_desc)
            u.anchor = (wL * aL + wR * aR) / (wL + wR)
        else:
            raise ValueError(f"Unknown merge='{merge}'")
        return u.anchor

    post(root)


def _beta_sample(a: float, b: float, device: torch.device) -> float:
    # Beta(a,b) via gamma
    x = torch._standard_gamma(torch.tensor([a], device=device))[0]
    y = torch._standard_gamma(torch.tensor([b], device=device))[0]
    return float((x / (x + y)).item())


def assign_event_times(
    root: Node,
    device: torch.device,
    split_beta: Tuple[float, float] = (1.0, 2.0),
    del_beta: Tuple[float, float] = (1.0, 1.0),
) -> None:
    a_s, b_s = split_beta
    a_d, b_d = del_beta

    def rec(u: Node, t0: float) -> None:
        if u.is_leaf():
            if u.leaf_deleted:
                u.del_time = t0 + (1.0 - t0) * _beta_sample(a_d, b_d, device)
            return
        u.split_time = t0 + (1.0 - t0) * _beta_sample(a_s, b_s, device)
        rec(u.left, u.split_time)
        rec(u.right, u.split_time)

    rec(root, 0.0)


def sigma_t(t: float, sigma0: float = 1.0, sigma1: float = 0.05) -> float:
    return (1.0 - t) * sigma0 + t * sigma1


def bridge_sample(x_s: torch.Tensor, s: float, v: float, anchor: torch.Tensor) -> torch.Tensor:
    if not (0.0 <= s <= v <= 1.0):
        raise ValueError("Require 0 <= s <= v <= 1")
    if s == 1.0:
        return anchor.clone()
    alpha = (v - s) / max(1e-8, (1.0 - s))
    sig = sigma_t(v)
    noise = sig * torch.randn_like(x_s)
    return (1.0 - alpha) * x_s + alpha * anchor + noise


@dataclass
class Particle:
    x: torch.Tensor
    tau: int
    b: Tuple[int, ...]   # 0=left/up, 1=right/down
    node: Node


def sample_path_single_variable(
    x1: torch.Tensor,              # [L] or [L,D]
    t: float,
    dr: float = 1.0,
    split_beta: Tuple[float, float] = (1.0, 2.0),
    del_beta: Tuple[float, float] = (1.0, 1.0),
    anchor_merge: str = "weighted_by_desc",
) -> Dict[str, Any]:
    """
    Returns variable-length tensors for a single example.
    (We will pad/stack across batch in the vectorized wrapper.)
    """
    device = x1.device
    L = int(x1.shape[0])

    # x0: single root
    x0_val = torch.randn_like(x1[0])

    # x1_plus: insert duplicates to create "to-be-deleted" leaves
    lam = max(0.0, L * dr - L)
    n_deleted = int(torch.poisson(torch.tensor([lam], device=device)).item()) if lam > 0 else 0

    x1_plus_list: List[torch.Tensor] = [x1[j].clone() for j in range(L)]
    for _ in range(n_deleted):
        pick = int(torch.randint(0, L, (1,), device=device).item())
        dup = x1[pick].clone()
        pos = int(torch.randint(0, len(x1_plus_list) + 1, (1,), device=device).item())
        x1_plus_list.insert(pos, dup)
    x1_plus = torch.stack(x1_plus_list, dim=0)

    # tree + anchors + event times
    root = sample_random_full_binary_tree(int(x1_plus.shape[0]))
    assign_leaf_deleted_flags(root, n_deleted=n_deleted)
    assign_anchors_bottom_up(root, x1_plus=x1_plus, merge=anchor_merge)
    assign_event_times(root, device=device, split_beta=split_beta, del_beta=del_beta)

    # evolve to time t
    def evolve(p: Particle, s: float, u: Node) -> List[Particle]:
        if u.is_leaf():
            if u.leaf_deleted and u.del_time is not None and t >= u.del_time:
                return []
            x_t_val = bridge_sample(p.x, s, t, u.anchor)
            return [Particle(x=x_t_val, tau=p.tau, b=p.b, node=u)]

        assert u.split_time is not None
        if t < u.split_time:
            x_t_val = bridge_sample(p.x, s, t, u.anchor)
            return [Particle(x=x_t_val, tau=p.tau, b=p.b, node=u)]

        x_split = bridge_sample(p.x, s, u.split_time, u.anchor)
        pL = Particle(x=x_split, tau=p.tau, b=p.b + (0,), node=u.left)
        pR = Particle(x=x_split, tau=p.tau, b=p.b + (1,), node=u.right)
        out: List[Particle] = []
        out.extend(evolve(pL, u.split_time, u.left))
        out.extend(evolve(pR, u.split_time, u.right))
        return out

    particles = evolve(Particle(x=x0_val, tau=1, b=tuple(), node=root), 0.0, root)

    # sort for stable ordering (leaf_idx if leaf else by path)
    def sort_key(pp: Particle):
        if pp.node.is_leaf():
            return (0, pp.node.leaf_idx if pp.node.leaf_idx is not None else 10**9)
        return (1, pp.b)

    particles.sort(key=sort_key)

    # pack variable-length tensors
    x_t = torch.stack([p.x for p in particles], dim=0)                 # [Lt] or [Lt,D]
    anchors_t = torch.stack([p.node.anchor for p in particles], dim=0) # [Lt] or [Lt,D]
    tau_t = torch.tensor([p.tau for p in particles], device=device, dtype=torch.long)  # [Lt]
    path_lens = torch.tensor([len(p.b) for p in particles], device=device, dtype=torch.long)  # [Lt]

    # paths as padded int vectors for this sample only (we'll repad across batch)
    max_depth = int(path_lens.max().item()) if particles else 0
    b_pad = torch.full((len(particles), max_depth), -1, device=device, dtype=torch.long)  # [Lt,depth]
    for i, p in enumerate(particles):
        if len(p.b) > 0:
            b_pad[i, : len(p.b)] = torch.tensor(p.b, device=device, dtype=torch.long)

    # targets
    R_target = torch.tensor([(p.node.n_desc - 1) for p in particles], device=device, dtype=torch.long)  # [Lt]
    rho_target = torch.tensor(
        [1 if (p.node.is_leaf() and p.node.leaf_deleted) else 0 for p in particles],
        device=device,
        dtype=torch.long,
    )  # [Lt]
    denoise_target = anchors_t.clone()  # x1-pred style target; fine for your separate compute_loss()

    return {
        "t": t,
        "x_t": x_t,
        "anchors_t": anchors_t,
        "tau_t": tau_t,
        "b_t": b_pad,
        "path_lens": path_lens,
        "R_target": R_target,
        "rho_target": rho_target,
        "denoise_target": denoise_target,
        "Lt": x_t.shape[0],
        "depth": max_depth,
    }


# ============================================================
# Vectorized output: dictionary of padded tensors + masks
# ============================================================

def sample_path(
    x1_batch: torch.Tensor,        # [B,L] or [B,L,D]
    t: Optional[torch.Tensor] = None,   # optional [B] tensor of times; else sampled
    dr: float = 1.0,
    split_beta: Tuple[float, float] = (1.0, 2.0),
    del_beta: Tuple[float, float] = (1.0, 1.0),
    anchor_merge: str = "weighted_by_desc",
) -> "BranchingFlowsInterpolantResult":
    """
    Returns a *dictionary of tensors* (padded to max Lt in batch), plus masks.

    Output shapes (scalar x):
      x_t          : [B, Lmax]
      anchors_t    : [B, Lmax]
      denoise_tgt  : [B, Lmax]
      R_target     : [B, Lmax]
      rho_target   : [B, Lmax]
      tau_t        : [B, Lmax]
      b_t          : [B, Lmax, Dmax]  (binary path, padded with -1)
      path_lens    : [B, Lmax]
      lt           : [B]
      mask         : [B, Lmax] (True where valid)

    For vector-valued x (x1: [B,L,D]), x_t/anchors/denoise_tgt become [B,Lmax,D].
    """
    device = x1_batch.device
    B = int(x1_batch.shape[0])

    # sample t if not provided
    if t is None:
        t = torch.rand((B,), device=device)
    else:
        assert t.shape == (B,), "t must be shape [B] if provided"

    # per-sample variable-length draws (hard to fully vectorize because trees vary),
    # then pad into a dictionary-of-tensors.
    per: List[Dict[str, Any]] = []
    Lmax = 0
    Dmax = 0
    for b in range(B):
        sb = sample_path_single_variable(
            x1_batch[b],
            t=float(t[b].item()),
            dr=dr,
            split_beta=split_beta,
            del_beta=del_beta,
            anchor_merge=anchor_merge,
        )
        per.append(sb)
        Lmax = max(Lmax, int(sb["Lt"]))
        Dmax = max(Dmax, int(sb["depth"]))

    # infer whether x is scalar or vector-valued
    x_is_vector = (per[0]["x_t"].dim() == 2)  # [Lt,D] vs [Lt]
    D = int(per[0]["x_t"].shape[-1]) if x_is_vector else 1

    # allocate padded tensors
    if x_is_vector:
        x_t = torch.zeros((B, Lmax, D), device=device, dtype=per[0]["x_t"].dtype)
        anchors_t = torch.zeros((B, Lmax, D), device=device, dtype=per[0]["anchors_t"].dtype)
        denoise_tgt = torch.zeros((B, Lmax, D), device=device, dtype=per[0]["denoise_target"].dtype)
    else:
        x_t = torch.zeros((B, Lmax), device=device, dtype=per[0]["x_t"].dtype)
        anchors_t = torch.zeros((B, Lmax), device=device, dtype=per[0]["anchors_t"].dtype)
        denoise_tgt = torch.zeros((B, Lmax), device=device, dtype=per[0]["denoise_target"].dtype)

    R_target = torch.zeros((B, Lmax), device=device, dtype=torch.long)
    rho_target = torch.zeros((B, Lmax), device=device, dtype=torch.long)
    tau_t = torch.zeros((B, Lmax), device=device, dtype=torch.long)
    path_lens = torch.zeros((B, Lmax), device=device, dtype=torch.long)
    b_t = torch.full((B, Lmax, Dmax), -1, device=device, dtype=torch.long)  # -1 = padding
    lt = torch.zeros((B,), device=device, dtype=torch.long)
    mask = torch.zeros((B, Lmax), device=device, dtype=torch.bool)

    # fill
    for b in range(B):
        sb = per[b]
        Lt = int(sb["Lt"])
        lt[b] = Lt
        mask[b, :Lt] = True

        if x_is_vector:
            x_t[b, :Lt, :] = sb["x_t"]
            anchors_t[b, :Lt, :] = sb["anchors_t"]
            denoise_tgt[b, :Lt, :] = sb["denoise_target"]
        else:
            x_t[b, :Lt] = sb["x_t"]
            anchors_t[b, :Lt] = sb["anchors_t"]
            denoise_tgt[b, :Lt] = sb["denoise_target"]

        R_target[b, :Lt] = sb["R_target"]
        rho_target[b, :Lt] = sb["rho_target"]
        tau_t[b, :Lt] = sb["tau_t"]
        path_lens[b, :Lt] = sb["path_lens"]

        depth = int(sb["depth"])
        if depth > 0:
            b_t[b, :Lt, :depth] = sb["b_t"]

    return BranchingFlowsInterpolantResult(
        t=t,
        x_t=x_t,
        anchors_t=anchors_t,
        g_tau=tau_t,
        g_b=b_t,
        g_b_lens=path_lens,
        targets_R=R_target,
        targets_rho=rho_target,
        targets_denoise=denoise_tgt,
        lt=lt,
        mask=mask,
    )


@dataclass
class BranchingFlowsInterpolantResult:
    t: Tensor                      # [B]
    x_t: Tensor                    # [B,Lmax] or [B,Lmax,D]
    anchors_t: Tensor              # [B,Lmax] or [B,Lmax,D]
    g_tau: Tensor                  # [B,Lmax]
    g_b: Tensor                    # [B,Lmax,Dmax]
    g_b_lens: Tensor               # [B,Lmax]
    targets_R: Tensor              # [B,Lmax]
    targets_rho: Tensor            # [B,Lmax]
    targets_denoise: Tensor        # [B,Lmax] or [B,Lmax,D]
    lt: Tensor                     # [B]
    mask: Tensor                   # [B,Lmax]


class BranchingFlowsInterpolant():
    def __init__(
        self,
        max_length: int,
        vocab_size: int = 10000,
        pad_token: int = 10001,
        bos_token: int = 10002,
        euclidean_dim: int = 3,
    ):
        super().__init__()
        self.max_length = max_length
        self.euclidean_dim = euclidean_dim
        self.vocab_size = vocab_size
        self.pad_token = pad_token
        self.bos_token = bos_token

    def dalpha(self, t):
        return torch.ones_like(t)

    def alpha(self, t):
        return t
    
    def sample_interpolant(self, t: Tensor, x1: Tensor, y1: Tensor,attn_mask: Tensor) -> BranchingFlowsInterpolantResult:
        return sample_path(
            x1_batch=x1,
            t=t,
            dr=1.2,
            split_beta=(1.0, 2.0),
            del_beta=(1.0, 1.0),
            anchor_merge="weighted_by_desc",
        )

    def jump_kernel_elbo(self, x, y, eps=1e-6):
        # x_safe: true length
        # y_safe: predicted length
        x_safe = torch.clamp(x, min=eps)
        y_safe = torch.clamp(y, min=eps)

        return y_safe - x_safe + x_safe * (torch.log(x_safe) - torch.log(y_safe))
 
    def sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        eps = 1e-5
        return torch.rand(batch_size, device=device) * (1 - eps)
    
    def compute_loss(self, model, batch):
        _x1 = batch["x"]
        _y1 = batch["y"]
        _mask_1 = batch["mask"]
        x1, y1, mask_1 = self.pad_sequence(_x1, _y1, _mask_1)
        t = self.sample_time(_x1.shape[0], _x1.device)
        interpolant_sample = self.sample_interpolant(t, x1, y1, mask_1)

        prediction: BranchingFlowsPrediction = model(
            euclidean_tokens=interpolant_sample.xt,
            cat_tokens=interpolant_sample.yt,
            symbols_mask=interpolant_sample.mask_t,
            pos_mask=interpolant_sample.mask_t,
            symbols_time=t,
            pos_time=t
        )
        mask_t_shaped = interpolant_sample.mask_t.unsqueeze(-1)

        masked_positions = (interpolant_sample.yt == self.mask_token)
        
        # Euclidean loss
        # We must only compute the loss for positions that are:
        # not deleted: mask_t_shaped
        # not masked: masked_positions.unsqueeze(-1)
        dsm_loss = (interpolant_sample.x1_ordered - prediction.clean_data)**2 * mask_t_shaped
        dsm_loss[:,0] = 0. # Don't take loss at the start of the sequence
        dsm_loss = dsm_loss.sum(dim=-1)[~masked_positions]
        dsm_loss = dsm_loss.mean() / x1.shape[-1]
        # Insertion loss
        insertion_rate = prediction.insertion_rate
        gaps, gaps_mask = interpolant_sample.gaps_and_mask
        insertion_loss = self.jump_kernel_elbo(gaps[gaps_mask], insertion_rate[gaps_mask])
        insertion_loss = insertion_loss.sum() / (y1.shape[0] * self.max_length) # This is not the best scaling factor
        # Unmasking loss
        # Reshape for cross_entropy: [batch, seq_len, num_classes] -> [batch * seq_len, num_classes]
        # and [batch, seq_len] -> [batch * seq_len]
        logits_flat = prediction.label_logits[masked_positions]
        targets_flat = interpolant_sample.y1_ordered[masked_positions]
        tokens_loss = F.cross_entropy(logits_flat, targets_flat, reduction="none").mean()

        # Predicted euclidean loss
        # Gather along V dimension: clean_data_unmasking is [B, L, V, D], y1 is [B, L] with vocab indices
        y1_indices = interpolant_sample.y1_ordered.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, x1.shape[-1])  # [B, L] -> [B, L, 1, D]
        predicted_cond_y1 = prediction.clean_data_unmasking.gather(dim=2, index=y1_indices).squeeze(2)  # [B, L, 1, D] -> [B, L, D]
        euclidean_loss = (predicted_cond_y1 - interpolant_sample.x1_ordered)**2
        euclidean_loss = euclidean_loss.sum(dim=-1)[masked_positions]
        euclidean_loss = euclidean_loss.mean() / x1.shape[-1]


        return {
            "dsm_loss": dsm_loss,
            "discrete_unmasking_loss": tokens_loss,
            "euclidean_unmasking_loss": euclidean_loss,
            "insertion_loss": insertion_loss,
        }
    
    def get_drift(self, prediction: BranchingFlowsPrediction, xt: Tensor, t: Tensor) -> Tensor:
        clean_data = prediction.clean_data 
        return (clean_data - xt) / (1 - self.alpha(t).view(-1, 1, 1))
    
    def get_insertion_rate(self, prediction: BranchingFlowsPrediction, t: Tensor) -> Tensor:
        coeff = self.dalpha(t) / (1 - self.alpha(t))
        rate = coeff.view(-1, 1) * prediction.insertion_rate
        return rate
    
    def get_unmasking_rate(self, prediction: BranchingFlowsPrediction, t: Tensor) -> Tensor:
        # Subtracting 4 because we are not considering the start of sequence token, end of sequence token, the mask token and the pad token
        coeff = self.dalpha(t) / (1 - self.alpha(t))
        rate = coeff.view(-1, 1, 1) * prediction.label_logits[:, :, :self.vocab_size-4].softmax(dim=-1)
        return rate
    
    def get_prior_distribution(self, batch_size: int, max_length: int, device: torch.device) -> Tensor:
        xt = torch.zeros((batch_size, max_length, self.euclidean_dim), device=device)
        yt = torch.ones((batch_size, max_length), device=device, dtype=torch.long) * self.pad_token
        yt[:,0] = self.bos_token
        mask_t = torch.zeros((batch_size, max_length), dtype=torch.bool, device=device)
        mask_t[:, 0] = True # Only the start is active
        return xt, yt, mask_t
    
    @torch.no_grad()
    def euclidean_sampling(
        self,
        model: torch.nn.Module,
        steps: int,
        batch_size: int,
        max_length: int,
        device: torch.device,
        return_trace: bool = False,
    ) -> SamplingResult:
        # 1) Initialize all‑pad sequence and trace
        xt, yt, mask_t = self.get_prior_distribution(batch_size, max_length, device)
        
        dt = 1.0 / steps
        t = torch.zeros(batch_size, device=device)

        trajectory = []
        if return_trace:
            trajectory.append(SamplingTrajectoryResult(
                xt=xt.clone(), yt=yt.clone(), mask_t=mask_t.clone(), t=t
            ))
        torch.set_printoptions(precision=2, sci_mode=False)
        for i in tqdm(range(steps), leave=False):
            prediction: BranchingFlowsPrediction = model(
                cat_tokens=yt,
                euclidean_tokens=xt,
                symbols_mask=mask_t,
                pos_mask=mask_t,
                symbols_time=t,
                pos_time=t
            )
            # Denoise
            masked_positions = (yt == self.mask_token)
            drift = self.get_drift(prediction, xt, t)
            xt = xt + (drift * dt)
            xt = torch.where(mask_t.unsqueeze(-1), xt, 0.)
            xt[:,0] = 0. # Don't corrupt begginning of the sequence
            xt = torch.where(masked_positions.unsqueeze(-1), 0., xt)

            # Unmasking
            unmasking_rate = self.get_unmasking_rate(prediction, t)
            unmasking_nums = torch.distributions.poisson.Poisson(unmasking_rate * dt).sample()
            num_jumps = unmasking_nums.sum(dim=-1)
            if i != steps - 1:
                change_pos = (num_jumps == 1) & (mask_t) & (masked_positions)
                unmasking_nums = unmasking_nums * change_pos.unsqueeze(-1)
                new_sample = unmasking_nums.argmax(dim=-1)
            else:
                change_pos = masked_positions
                new_sample = unmasking_rate.argmax(dim=-1)
            
            indices = new_sample.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, xt.shape[-1])  # [B, L] -> [B, L, 1, D]
            mean_cond_y1 = prediction.clean_data_unmasking.gather(dim=2, index=indices).squeeze(2)

            yt = torch.where(change_pos, new_sample, yt)
            new_xt = self.alpha(t).view(-1, 1, 1) * mean_cond_y1 + (1 - self.alpha(t).view(-1, 1, 1)) * torch.randn_like(xt)
            xt = torch.where(change_pos.unsqueeze(-1), new_xt, xt)


            # Perform insertions
            insertion_rate = self.get_insertion_rate(prediction, t)
            ext = torch.distributions.poisson.Poisson(insertion_rate * dt).sample()

            seq_len = xt.shape[1]
            if i != steps - 1:
                for j in range(batch_size):
                    # Add dimensions
                    new_xt_j = []
                    new_yt_j = []
                    for k in range(seq_len):
                        if not mask_t[j, k]:
                            break
                        new_xt_j.append(xt[j, k:k+1, :])
                        new_yt_j.append(yt[j, k].item())
                        # Insert new token after position k if ext[j, k] > 0
                        if ext[j, k] > 0:
                            # Consider doing a for loop here
                            # print('------------- Position ', k, ' -------------')
                            for _ in range(int(ext[j, k].item())):
                                new_sample = torch.zeros_like(xt[0, :1, :])
                                new_xt_j.append(new_sample.clone())
                                new_yt_j.append(self.mask_token)
                    
                    # Ensure we don't exceed the tensor size
                    new_len = min(len(new_xt_j), max_length) 
                    new_xt_tensor = torch.cat(new_xt_j[:new_len], dim=0)
                    new_yt_tensor = torch.tensor(new_yt_j[:new_len], device=device, dtype=yt.dtype)
                    xt[j, :new_len, :] = new_xt_tensor
                    yt[j, :new_len] = new_yt_tensor
                    xt[j, 0] = 0.
                    yt[j, 0] = self.bos_token
                    mask_t[j, :new_len] = True
                    mask_t[j, new_len:] = False

            if return_trace:
                trajectory.append(SamplingTrajectoryResult(
                    xt=xt.clone(), yt=yt.clone(), mask_t=mask_t.clone(), t=t
                ))
            t = t + dt

        return SamplingResult(
            xt=xt, yt=yt, mask_t=mask_t, trajectory=trajectory
        )

def sample_categorical(categorical_probs, method="hard"):
    if method == "hard":
        gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
        return (categorical_probs / gumbel_norm).argmax(dim=-1)
    else:
        raise ValueError(f"Method {method} for sampling categorical variables is not valid.")


# ============================================================
# Example
# ============================================================

if __name__ == "__main__":
    device = torch.device("cpu")

    B, L = 4, 5
    x1 = torch.randn(B, L, device=device)

    out = sample_path(
        x1_batch=x1,
        dr=1.2,
        split_beta=(1.0, 2.0),
        del_beta=(1.0, 1.0),
        anchor_merge="weighted_by_desc",
    )

    for name, value in out.__dict__.items():
        if isinstance(value, torch.Tensor):
            print(name, tuple(value.shape), value.dtype)
        else:
            print(name, value)
