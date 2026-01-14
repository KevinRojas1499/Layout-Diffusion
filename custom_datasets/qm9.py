import torch
import datasets
import numpy as np
from torch.utils.data import Dataset
from utils.tokenizer import VocabTokenizer

class QM9Dataset(Dataset):
    def __init__(self, tokenizer: VocabTokenizer, max_length=30, canonical_order=True):
        self.qm9_dataset = datasets.load_dataset(
            'yairschiff/qm9',
            split='train')  # Dataset only has 'train' split

        self.max_length = max_length
        self.tokenizer = tokenizer
        self.canonical_order = canonical_order
        
        # Atomic number mapping for canonical ordering
        self.atomic_numbers = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5,
            'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
            'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
            'S': 16, 'Cl': 17, 'Ar': 18
        }

    def _canonical_order_atoms(self, atomic_symbols, pos):
        """
        Order atoms in a canonical way:
        1. By atomic number
        2. By distance from centroid (for same atomic number)
        3. By x, y, z coordinates (for same atomic number and distance)
        """
        if len(atomic_symbols) == 0:
            return atomic_symbols, pos
        
        # Convert to numpy arrays for easier manipulation
        atomic_symbols = list(atomic_symbols)
        pos = np.array(pos)
        
        # Calculate centroid
        centroid = np.mean(pos, axis=0)
        
        # Calculate distances from centroid
        distances = np.linalg.norm(pos - centroid, axis=1)
        
        # Create list of tuples for sorting: (atomic_number, distance, x, y, z, original_index)
        sort_keys = []
        for i, (symbol, dist, p) in enumerate(zip(atomic_symbols, distances, pos)):
            atomic_num = self.atomic_numbers.get(symbol, 99)  # Default to 99 for unknown elements
            sort_keys.append((atomic_num, dist, p[0], p[1], p[2], i))
        
        # Sort by the keys
        sorted_indices = sorted(range(len(sort_keys)), key=lambda i: sort_keys[i])
        
        # Reorder atomic_symbols and pos
        ordered_symbols = [atomic_symbols[i] for i in sorted_indices]
        ordered_pos = pos[sorted_indices].tolist()
        
        return ordered_symbols, ordered_pos

    def __len__(self):
        return len(self.qm9_dataset)

    def __getitem__(self, index):
        data = self.qm9_dataset[index]
        atomic_symbols = data['atomic_symbols']
        pos = data['pos']
        
        # Apply canonical ordering if requested
        if self.canonical_order:
            atomic_symbols, pos = self._canonical_order_atoms(atomic_symbols, pos)
        
        # Pad the atomic symbols and pos to the max length
        original_length = len(pos)
        pos = pos + [[0, 0, 0]] * (self.max_length - len(pos))
        pos = torch.tensor(pos, dtype=torch.float32)

        # Tokenize and pad the atomic symbols
        atomic_symbols = self.tokenizer.pad_tokenize(atomic_symbols, self.max_length)

        mask = torch.ones(self.max_length, dtype=torch.bool)
        mask[original_length:] = False
        return {'x': pos, 'y': atomic_symbols, 'mask': mask}

