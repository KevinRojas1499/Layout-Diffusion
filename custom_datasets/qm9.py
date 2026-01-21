import torch
import datasets
import numpy as np
from torch.utils.data import Dataset
from utils.tokenizer import VocabTokenizer
from rdkit import Chem

class QM9Dataset(Dataset):
    def __init__(self, tokenizer: VocabTokenizer, max_length=30, canonical_order=True):
        self.qm9_dataset = datasets.load_from_disk(
            'data/qm9_preprocessed'
        )

        self.max_length = max_length
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.qm9_dataset)

    def __getitem__(self, index):
        data = self.qm9_dataset[index]
        atomic_symbols = data['atomic_symbols']
        pos = data['pos']
        
        # Pad the atomic symbols and pos to the max length
        original_length = len(pos)
        pos = pos + [[0, 0, 0]] * (self.max_length - len(pos))
        pos = torch.tensor(pos, dtype=torch.float32)

        # Tokenize and pad the atomic symbols
        atomic_symbols = self.tokenizer.pad_tokenize(atomic_symbols, self.max_length)

        mask = torch.ones(self.max_length, dtype=torch.bool)
        mask[original_length:] = False
        return {'x': pos, 'y': atomic_symbols, 'mask': mask}

