from __future__ import annotations
import abc
import random
import torch
from torch import Tensor
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Callable
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm


@dataclass
class BranchingFlowsPrediction:
    # This handles denoising
    clean_data: Tensor
    # This handles the unmasking
    label_logits: Tensor
    split_rates: Tensor

@dataclass
class BranchingFlowsInterpolantResult:
    xt: Tensor # Shape [Batch, Length]
    yt: Tensor # Shape [Batch, Length]
    mask_t: Tensor # Shape [Batch, Length]
    t: Tensor # Shape [Batch]
    split_rates: Tensor # Shape [Batch, Length]
    denoising_target: Tensor # Shape [Batch, Length, D]
    discrete_target: Tensor # Shape [Batch, Length]
    # deletion_rates: Tensor # Shape [Batch, Length]

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


class DataPoint(abc.ABC):
    def __init__(self):
        super().__init__()
    
    @abc.abstractmethod
    def fuse(self, other : DataPoint) -> DataPoint:
        pass

class MultimodalDataPoint(DataPoint):
    def __init__(self, x : Tensor, y : Tensor, mask_token : int):
        super().__init__()
        self.x = x
        self.y = y
        self.mask_token = mask_token
    
    def __str__(self) -> str:
        return f"x={self.x}, y={self.y}"
    
    def fuse(self, other : MultimodalDataPoint) -> MultimodalDataPoint:
        new_x = (self.x  + other.x) / 2
        new_y = torch.ones_like(self.y) * self.mask_token
        return MultimodalDataPoint(
            x = new_x,
            y = new_y,
            mask_token = self.mask_token
        )

class Node:
    def __init__(self, data_point : Optional[DataPoint] = None):
        super().__init__()
        self.right = None
        self.left = None
        self.data_point = data_point
        
        self.computed_subtree_size = False
        self.subtree_size = 1

        self.splitting_time = None
    
    def is_leaf(self) -> bool:
        return self.right is None and self.left is None
    # Here I define subtree size as the number of leaves in the subtree
    def get_subtree_size(self) -> int:
        if self.computed_subtree_size:
            return self.subtree_size
        
        if self.is_leaf():
            self.computed_subtree_size = True
            self.subtree_size = 1
            return 1
        else:
            subtree_size = 0
            if self.left is not None:
                subtree_size += self.left.get_subtree_size()
            if self.right is not None:
                subtree_size += self.right.get_subtree_size()
            self.computed_subtree_size = True
            self.subtree_size = subtree_size
            return subtree_size
        
    def _sort_nodes(self) -> List[Node]:
        if self.is_leaf():
            return [self]
        else:
            return self.left._sort_nodes() + self.right._sort_nodes() + [self]
       
    def __str__(self) -> str:
        lines = []

        def format_data_point(node: Node) -> str:
            dp = node.data_point
            if dp is None:
                return ""
            if isinstance(dp, MultimodalDataPoint):
                x = dp.x
                y = dp.y
                x_str = f"x={x}" if hasattr(x, "shape") else f"x={x}"
                if hasattr(y, "shape"):
                    if y.numel() == 1:
                        y_str = f"y={int(y.item())}"
                    else:
                        y_str = f"y={y}"
                else:
                    y_str = f"y={y}"
                return f" [{x_str}, {y_str}, split_time={node.splitting_time}]"
            return f" [{dp}]"

        def rec(node, prefix: str, is_tail: bool, label: str) -> None:
            if node is None:
                return
            data_str = format_data_point(node)
            head = f"{label}({node.get_subtree_size()}).{data_str}" if label else f".{data_str}"
            if prefix == "":
                lines.append(head)
            else:
                connector = "└── " if is_tail else "├── "
                lines.append(prefix + connector + head)

            children = [("L:", node.left), ("R:", node.right)]
            present = [(lab, child) for lab, child in children if child is not None]
            for i, (lab, child) in enumerate(present):
                last = i == len(present) - 1
                child_prefix = prefix + ("    " if is_tail else "│   ")
                rec(child, child_prefix, last, lab)

        rec(self, "", True, "")
        return "\n".join(lines)


class RandomTree:
    def __init__(self, size : int, leaves : List[DataPoint]) -> None:
        super().__init__()
        self.size = size
        self.root = Node(None)
        self._construct_tree()
        self.root.get_subtree_size()
        self._compute_splitting_time()
        self._compute_anchors(leaves)
    
    def _construct_tree(self, ):
        leaf_nodes = [self.root]
        node_to_idx = {self.root: 0}
        while len(leaf_nodes) < self.size:
            # Pick a random leaf node
            node = random.choice(leaf_nodes)
            node_idx = node_to_idx[node]
            # Remove that node from the active list
            last = leaf_nodes[-1]
            leaf_nodes[node_idx] = last
            leaf_nodes.pop()
            node_to_idx[last] = node_idx
            node_to_idx.pop(node)
            # Add two new nodes as children of the removed node
            node.left = Node()
            node.right = Node()
            leaf_nodes.append(node.left)
            node_to_idx[node.left] = len(leaf_nodes) - 1
            leaf_nodes.append(node.right)
            node_to_idx[node.right] = len(leaf_nodes) - 1
    
    def _compute_splitting_time(self, node : Node = None, parent_time : float = 0.0) -> None:
        if node is None:
            node = self.root
        if node.is_leaf():
            node.splitting_time = parent_time
        else:
            u = random.random()
            splitting_time = 1 - u**(1 / (node.get_subtree_size() - 1)) * (1 - parent_time)
            node.splitting_time = splitting_time
            self._compute_splitting_time(node.left, splitting_time)
            self._compute_splitting_time(node.right, splitting_time)


    def _compute_anchors(self, leave_values) -> None:
        # Leave values is an indexable of size [self.size]
        sorted_nodes = self.root._sort_nodes()
        leaf_index = 0
        for i, node in enumerate(sorted_nodes):
            if node.right is None and node.left is None:
                # Node is a leaf
                node.data_point = leave_values[leaf_index]
                leaf_index += 1
            else:
                node.data_point = node.left.data_point.fuse(node.right.data_point)

    def get_data_at_t(self, t: float) -> Tuple[List[DataPoint], List[int]]:
        node = self.root
        data = []
        tree_sizes = []
        def construct_active_anchors(node: Node, t: float):
            if node.is_leaf():
                tree_sizes.append(node.get_subtree_size())
                data.append(node.data_point)
            else:
                if t < node.splitting_time:
                    tree_sizes.append(node.get_subtree_size())
                    data.append(node.data_point)
                else:
                    construct_active_anchors(node.left, t)
                    construct_active_anchors(node.right, t)
        construct_active_anchors(node, t)
        return data, tree_sizes
        
    def __str__(self) -> str:
        return str(self.root)


def _layout_tree_positions(root: Node) -> Dict[Node, Tuple[float, float]]:
    positions: Dict[Node, Tuple[float, float]] = {}
    next_x = 0

    def assign(node: Node, depth: int) -> float:
        nonlocal next_x
        if node.is_leaf():
            x = float(next_x)
            next_x += 1
        else:
            left_x = assign(node.left, depth + 1) if node.left is not None else None
            right_x = assign(node.right, depth + 1) if node.right is not None else None
            if left_x is None:
                x = right_x
            elif right_x is None:
                x = left_x
            else:
                x = (left_x + right_x) / 2.0
        positions[node] = (x, -float(depth))
        return x

    assign(root, 0)
    return positions


def plot_tree(root: Node, out_path: Optional[str] = None, show: bool = False) -> None:
    positions = _layout_tree_positions(root)
    fig, ax = plt.subplots(figsize=(8, 4))

    def format_tensor(value: Tensor) -> str:
        if value.numel() == 1:
            return f"{float(value.item()):.3f}"
        if value.ndim == 1 and value.numel() <= 4:
            vals = ", ".join(f"{float(v):.2f}" for v in value.tolist())
            return f"[{vals}]"
        return f"shape={tuple(value.shape)}"

    def format_data_point(node: Node) -> str:
        dp = node.data_point
        if dp is None:
            return ""
        if isinstance(dp, MultimodalDataPoint):
            x_str = format_tensor(dp.x) if isinstance(dp.x, torch.Tensor) else str(dp.x)
            y_str = format_tensor(dp.y) if isinstance(dp.y, torch.Tensor) else str(dp.y)
            return f"x={x_str}\ny={y_str}"
        return str(dp)

    for node, (x, y) in positions.items():
        if node.left is not None:
            x2, y2 = positions[node.left]
            ax.plot([x, x2], [y, y2], color="black", linewidth=1)
        if node.right is not None:
            x2, y2 = positions[node.right]
            ax.plot([x, x2], [y, y2], color="black", linewidth=1)

    for node, (x, y) in positions.items():
        ax.scatter([x], [y], s=180, color="white", edgecolor="black", zorder=3)
        split_str = (
            f"t={node.splitting_time:.2f}"
            if node.splitting_time is not None
            else "t=?"
        )
        data_str = format_data_point(node)
        label = f"{split_str}\n{data_str}" if data_str else split_str
        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none"),
            zorder=4,
        )

    ax.set_axis_off()
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    
class BranchingFlowsInterpolant():
    def __init__(self, mask_token: int, pad_token: int, vocab_size: int, euclidean_dim: int):
        self.mask_token = mask_token
        self.pad_token = pad_token
        self.vocab_size = vocab_size
        self.euclidean_dim = euclidean_dim
    
    def alpha(self, t: Tensor) -> Tensor:
        return t

    def dalpha(self, t: Tensor) -> Tensor:
        return torch.ones_like(t)
    
    def sample_interpolant(self, t: Tensor, x1: Tensor, y1: Tensor, attn_mask: Tensor):
        lengths = attn_mask.sum(dim=1)

        x1s = []
        y1s = []
        masks = []
        split_rates = []
        for i in range(x1.shape[0]):
            # Branching flows can't be interpolated that well
            leaves = [MultimodalDataPoint(x1[i,j, :], y1[i,j], self.mask_token) for j in range(lengths[i])]
            tree = RandomTree(lengths[i], leaves)

            # Extract the data at time t
            anchors, tree_sizes = tree.get_data_at_t(t[i])
            data_x = torch.stack([anchor.x for anchor in anchors])
            data_y = torch.stack([anchor.y for anchor in anchors])
            split_rates_i = torch.tensor(tree_sizes, device=x1.device)

            # Pad the data to the same length
            num_pads = x1.shape[1] - data_x.shape[0]
            mask = torch.cat((torch.ones(data_x.shape[0], device=x1.device, dtype=torch.bool), 
                             torch.zeros(num_pads, device=x1.device, dtype=torch.bool))
                             )
            data_x = torch.cat((data_x, torch.zeros(num_pads, data_x.shape[1], device=x1.device)))
            data_y = torch.cat((data_y, torch.ones(num_pads, device=x1.device, dtype=torch.long) * self.pad_token))
            split_rates_i = torch.cat((split_rates_i, torch.zeros(num_pads, device=x1.device, dtype=torch.float32)))


            x1s.append(data_x)
            y1s.append(data_y)
            masks.append(mask)
            split_rates.append(split_rates_i)
        
        x1s = torch.stack(x1s)
        y1s = torch.stack(y1s)
        mask_t = torch.stack(masks)
        split_rates = torch.stack(split_rates)
        # Euclidean data
        xt = t.view(-1,1,1) * x1s + (1 - t.view(-1,1,1)) * torch.randn_like(x1s)
        xt = torch.where(mask_t.unsqueeze(-1), xt, 0.)
        # Discrete data
        random_vals = torch.randint_like(y1s, 0, self.vocab_size)
        pos_to_change = (t.unsqueeze(-1) <= torch.rand_like(y1s,dtype=torch.float32))
        yt = torch.where(pos_to_change, random_vals, y1s) 

        return BranchingFlowsInterpolantResult(
            xt=xt,
            yt=yt,
            mask_t=mask_t,
            t=t,
            split_rates=split_rates,
            denoising_target=x1s,
            discrete_target=y1s,
        )

    def sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        eps = 1e-5
        return torch.rand(batch_size, device=device) * (1 - eps)

    def jump_kernel_elbo(self, x, y, eps=1e-6):
        # x_safe: true length
        # y_safe: predicted length
        x_safe = torch.clamp(x, min=eps)
        y_safe = torch.clamp(y, min=eps)

        return y_safe - x_safe + x_safe * (torch.log(x_safe) - torch.log(y_safe))
    def compute_loss(self, model, batch):
        x1 = batch["x"]
        y1 = batch["y"]
        mask_1 = batch["mask"]
        t = self.sample_time(x1.shape[0], x1.device)
        interpolant_sample = self.sample_interpolant(t, x1, y1, mask_1)

        prediction: BranchingFlowsPrediction = model(
            euclidean_tokens=interpolant_sample.xt,
            cat_tokens=interpolant_sample.yt,
            symbols_mask=interpolant_sample.mask_t,
            pos_mask=interpolant_sample.mask_t,
            symbols_time=t,
            pos_time=t
        )
        mask_t = interpolant_sample.mask_t

        # Euclidean loss
        dsm_loss = (interpolant_sample.denoising_target - prediction.clean_data)**2 * mask_t.unsqueeze(-1)
        dsm_loss = dsm_loss.sum() / mask_t.sum()

        # Insertion loss
        insertion_loss = self.jump_kernel_elbo(interpolant_sample.split_rates[mask_t], prediction.split_rates[mask_t])
        insertion_loss = insertion_loss.sum() / mask_t.sum()

        # Unmasking loss
        # Reshape for cross_entropy: [batch, seq_len, num_classes] -> [batch * seq_len, num_classes]
        # and [batch, seq_len] -> [batch * seq_len]
        logits_flat = prediction.label_logits[mask_t]
        targets_flat = interpolant_sample.discrete_target[mask_t]
        tokens_loss = F.cross_entropy(logits_flat, targets_flat, reduction="none").mean()

        return {
            "dsm_loss": dsm_loss,
            "discrete_unmasking_loss": tokens_loss,
            "insertion_loss": insertion_loss,
        }

    def get_prior_distribution(self, batch_size: int, max_length: int, device: torch.device) -> Tuple[Tensor, Tensor, Tensor]:
        xt = torch.randn((batch_size, max_length, self.euclidean_dim), device=device)
        xt[:, 1:] = 0.
        yt = torch.ones((batch_size, max_length), device=device, dtype=torch.long) * self.pad_token
        yt[:,0] = self.mask_token
        mask_t = torch.zeros((batch_size, max_length), dtype=torch.bool, device=device)
        mask_t[:, 0] = True # Only the start is active
        return xt, yt, mask_t
    
    def get_split_rates(self, prediction: BranchingFlowsPrediction, t: Tensor) -> Tensor:
        alpha = self.alpha(t)
        return prediction.split_rates / (1 - alpha).view(-1, 1) # normally woul have dalpha as well
    
    def get_denoising_rates(self, prediction: BranchingFlowsPrediction, t: Tensor) -> Tensor:
        alpha = self.alpha(t)
        prob = prediction.label_logits.softmax(dim=-1) 
        return prob / (1 - alpha).view(-1, 1, 1)

    def get_drift(self, prediction: BranchingFlowsPrediction, xt: Tensor, t: Tensor) -> Tensor:
        clean_data = prediction.clean_data 
        return (clean_data - xt) / (1 - self.alpha(t).view(-1, 1, 1))

    @torch.no_grad()
    def sampling(
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

            # Unmasking
            denoising_rate = self.get_denoising_rates(prediction, t)
            denoising_nums = torch.distributions.poisson.Poisson(denoising_rate * dt).sample()
            num_jumps = denoising_nums.sum(dim=-1)
            if i != steps - 1:
                change_pos = (num_jumps == 1) & (mask_t) & (masked_positions)
                denoising_nums = denoising_nums * change_pos.unsqueeze(-1)
                new_sample = denoising_nums.argmax(dim=-1)
            else:
                # verify this is correct
                change_pos = masked_positions
                new_sample = denoising_rate.argmax(dim=-1)
            
            yt = torch.where(change_pos, new_sample, yt)
            
            # Perform insertions
            split_rates = self.get_split_rates(prediction, t)
            ext = torch.distributions.poisson.Poisson(split_rates * dt).sample()

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
                            new_xt_j.append(xt[j, k:k+1, :].clone())
                            new_yt_j.append(yt[j, k].item())
                    
                    # Ensure we don't exceed the tensor size
                    new_len = min(len(new_xt_j), max_length) 
                    new_xt_tensor = torch.cat(new_xt_j[:new_len], dim=0)
                    new_yt_tensor = torch.tensor(new_yt_j[:new_len], device=device, dtype=yt.dtype)
                    xt[j, :new_len, :] = new_xt_tensor
                    yt[j, :new_len] = new_yt_tensor
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