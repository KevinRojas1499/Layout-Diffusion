import torch
import datasets
from torch.utils.data import Dataset
from utils.tokenizer import VocabTokenizer

class QM9Dataset(Dataset):
    def __init__(self, tokenizer: VocabTokenizer, max_length=30):
        self.qm9_dataset = datasets.load_dataset(
            'yairschiff/qm9',
            split='train')  # Dataset only has 'train' split

        self.max_length = max_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.qm9_dataset)

    def __getitem__(self, index):
        data = self.qm9_dataset[index]
        atomic_symbols = data['atomic_symbols']
        pos = data['pos']
        # Pad the atomic symbols and pos to the max length
        pos = pos + [[0, 0, 0]] * (self.max_length - len(pos))

        # Tokenize and pad the atomic symbols
        atomic_symbols = self.tokenizer.pad_tokenize(atomic_symbols, self.max_length)

        mask = torch.ones(self.max_length, dtype=torch.bool)
        mask[len(atomic_symbols):] = False
        return {'x': pos, 'y': atomic_symbols, 'mask': mask}

