from __future__ import annotations
import abc
import random
import torch
from torch import Tensor
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Callable
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


class DataPoint(abc.ABC):
    def __init__(self):
        super().__init__()
    
    @abc.abstractmethod
    def merge_data(self, points : List[DataPoint]):
        pass
    
    @abc.abstractmethod
    def fuse(self, other : DataPoint) -> DataPoint:
        pass

class MultimodalDataPoint(DataPoint):
    def __init__(self, x : Tensor, y : Tensor, mask_token : int):
        super().__init__()
        print('creating point' , x, y, mask_token)
        self.x = x
        self.y = y
        self.mask_token = mask_token
    
    def __str__(self) -> str:
        return f"x={self.x}, y={self.y}, mask_token={self.mask_token}"
    
    def merge_data(self,  points : MultimodalDataPoint) -> MultimodalDataPoint:
        return {
            'x', torch.stack([point.x for point in points]), 
            'y', torch.cat([point.y for point in points], dim=0), 
            'mask', torch.cat([point.mask for point in points], dim=0)
        }
    
    def fuse(self, other : MultimodalDataPoint) -> MultimodalDataPoint:
        new_x = (self.x  + other.x) / 2
        new_y = self.mask_token
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
    
    # Here I define subtree size as the number of leaves in the subtree
    def get_subtree_size(self) -> int:
        if self.computed_subtree_size:
            return self.subtree_size
        
        if self.right is None and self.left is None:
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
        if self.right is None and self.left is None:
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
                return f" [{x_str}, {y_str}]"
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
    
    def _compute_splitting_time(self, node : Node) -> None:
        pass


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

    def __str__(self) -> str:
        return str(self.root)
    

n = 5 # Number of leaves
x = torch.arange(n, dtype=torch.float32)
y = torch.arange(n)
mask_token = 11

leaves = [MultimodalDataPoint(x, y, mask_token) for x, y in zip(x, y)]
print([str(leaf) for leaf in leaves])
tree = RandomTree(n, leaves)
print(tree)

            
                