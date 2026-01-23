import torch
import datasets
import numpy as np
from torch.utils.data import Dataset
from utils.tokenizer import VocabTokenizer

class QM9Dataset(Dataset):
    def __init__(self, tokenizer: VocabTokenizer, max_length=30, use_raw_dataset=True):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.use_raw_dataset = use_raw_dataset

        self.qm9_dataset = datasets.load_from_disk(
            'data/qm9_preprocessed'
        )
        
    def __len__(self):
        return len(self.qm9_dataset)

    def __getitem__(self, index):
        data = self.qm9_dataset[index]
        if self.use_raw_dataset:
            atomic_symbols = data['original_atomic_symbols']
            pos = data['original_pos']
        else:
            atomic_symbols = data['atomic_symbols']
            pos = data['pos']
            atomic_symbols, pos = self.reorder_hydrogens_only(atomic_symbols, pos)

        
        # Pad the atomic symbols and pos to the max length
        original_length = len(pos)
        pos = np.concatenate([pos, [[0, 0, 0]] * (self.max_length - len(pos))])
        pos = torch.from_numpy(pos).float()

        # Tokenize and pad the atomic symbols
        atomic_symbols = self.tokenizer.pad_tokenize(atomic_symbols, self.max_length)

        mask = torch.ones(self.max_length, dtype=torch.bool)
        mask[original_length:] = False
        return {'x': pos, 'y': atomic_symbols, 'mask': mask}

