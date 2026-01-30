import re
import torch
import datasets
import numpy as np
from torch.utils.data import Dataset
from utils.tokenizer import VocabTokenizer

class EquationsDataset(Dataset):
    def __init__(self, tokenizer: VocabTokenizer, max_length=8, data_path='data/equations_varlen.txt'):
        self.max_length = max_length
        self.tokenizer = tokenizer

        self.equations_list = [equation.strip() for equation in open(data_path, 'r').readlines()]


    def __len__(self):
        return len(self.equations_list)

    def __getitem__(self, index):
        data = self.equations_list[index]
        tokens = []
        i = 0
        while i < len(data):
            ch = data[i]
            if ch.isspace():
                i += 1
                continue
            if ch.isdigit():
                j = i + 1
                while j < len(data) and data[j].isdigit():
                    j += 1
                tokens.append(data[i:j])
                i = j
                continue
            if ch == '-':
                prev = tokens[-1] if tokens else None
                next_is_digit = i + 1 < len(data) and data[i + 1].isdigit()
                if next_is_digit and (prev is None or prev in ['+', '-', '*', '/', '=', '.']):
                    j = i + 1
                    while j < len(data) and data[j].isdigit():
                        j += 1
                    tokens.append(data[i:j])
                    i = j
                else:
                    tokens.append(ch)
                    i += 1
                continue
            if ch in ['+', '*', '/', '=', '.']:
                tokens.append(ch)
                i += 1
                continue
            raise ValueError(f"Unexpected character in equation: {ch}")

        numbers = [int(token) for token in tokens if re.fullmatch(r'-?\d+', token)]
        symbols = [token for token in tokens if token in self.tokenizer.vocab]
        numbers = torch.tensor(numbers, dtype=torch.float32).unsqueeze(-1)
        length = len(numbers)
        symbols = self.tokenizer.pad_tokenize(symbols, self.max_length)
        numbers = torch.cat([numbers, torch.zeros(self.max_length - length, 1)], dim=0)

        mask = torch.ones(self.max_length, dtype=torch.bool)
        mask[length:] = False
        return {'x': numbers, 'y': symbols, 'mask': mask}

