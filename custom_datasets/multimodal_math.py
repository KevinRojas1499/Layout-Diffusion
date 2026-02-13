"""Dataset for arithmetic equations stored in JSONL format."""

import json
import torch
from torch.utils.data import Dataset
from utils.tokenizer import VocabTokenizer


class EquationsDataset(Dataset):
    """Load pre-parsed equations from JSONL.
    
    Each sample returns:
        - x: numbers tensor, shape [max_length, 1]
        - y: tokenized symbols, shape [max_length]
        - mask: valid positions, shape [max_length]
    """
    
    def __init__(
        self,
        tokenizer: VocabTokenizer,
        max_length: int = 8,
        data_path: str = "data/equations_l5.jsonl",
    ):
        self.max_length = max_length
        self.tokenizer = tokenizer
        
        self.data = []
        with open(data_path) as f:
            for line in f:
                item = json.loads(line)
                self.data.append({
                    "numbers": item["numbers"],
                    "symbols": item["symbols"],
                })

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.data[idx]
        length = len(item["numbers"])
        
        numbers = torch.tensor(item["numbers"], dtype=torch.float32).unsqueeze(-1)
        symbols = self.tokenizer.pad_tokenize(item["symbols"], self.max_length)
        
        # Pad numbers
        if length < self.max_length:
            numbers = torch.cat([numbers, torch.zeros(self.max_length - length, 1)])
        
        mask = torch.zeros(self.max_length, dtype=torch.bool)
        mask[:length] = True
        
        return {"x": numbers, "y": symbols, "mask": mask}
