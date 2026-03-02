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
        self.max_length = 0
        with open(data_path) as f:
            for line in f:
                item = json.loads(line)
                self.max_length = max(self.max_length, len(item["symbols"]))
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


class ParenthesizedEquationsDataset(Dataset):
    """Dataset variant for equations that include parentheses tokens in symbols.

    This keeps parentheses in the categorical sequence (`y`) and aligns numeric
    values (`x`) to operator/anchor symbol positions.
    """

    def __init__(
        self,
        tokenizer: VocabTokenizer,
        data_path: str = "data/equations_l5.jsonl",
    ):
        self.tokenizer = tokenizer

        self.data = []
        self.max_length = 0
        with open(data_path) as f:
            for line in f:
                item = json.loads(line)
                self.max_length = max(self.max_length, len(item["symbols"]))
                self.data.append(
                    {
                        "numbers": item["numbers"],
                        "symbols": item["symbols"],
                    }
                )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.data[idx]
        symbols_raw: list[str] = item["symbols"]
        numbers_raw: list[float] = item["numbers"]
        y = self.tokenizer.pad_tokenize(symbols_raw, self.max_length)

        x = torch.zeros(self.max_length, 1, dtype=torch.float32)
        x[:len(numbers_raw), 0] = torch.tensor(numbers_raw, dtype=torch.float32)

        mask_y = torch.zeros(self.max_length, dtype=torch.bool)
        mask_y[:len(symbols_raw)] = True

        mask_x = torch.zeros(self.max_length, dtype=torch.bool)
        mask_x[:len(numbers_raw)] = True

        return {
            "x": x,
            "y": y,
            "mask_x": mask_x,
            "mask_y": mask_y,
            # Compatibility for codepaths expecting a single mask.
            "mask": mask_y,
        }
