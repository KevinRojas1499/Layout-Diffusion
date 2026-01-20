import torch
import datasets
import numpy as np
from torch.utils.data import Dataset
from utils.tokenizer import VocabTokenizer
from rdkit import Chem

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

    def _smiles_based_order_atoms(self, atomic_symbols, pos, smiles):
        """
        Order atoms to match canonical SMILES string ordering.
        Heavy atoms come first (in SMILES order), then hydrogens.
        This matches the paper's approach.
        """
        if len(atomic_symbols) == 0:
            return atomic_symbols, pos
        
        # Parse SMILES to get heavy atom order
        mol = Chem.MolFromSmiles(smiles)
        # Get heavy atom symbols in SMILES order
        smiles_heavy_atoms = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(mol.GetNumAtoms())]
        
        # Separate heavy atoms and hydrogens from the actual molecule
        atomic_symbols = list(atomic_symbols)
        pos = np.array(pos)
        
        heavy_indices = []
        heavy_symbols = []
        heavy_positions = []
        hydrogen_indices = []
        hydrogen_symbols = []
        hydrogen_positions = []
        
        for i, symbol in enumerate(atomic_symbols):
            if symbol == 'H':
                hydrogen_indices.append(i)
                hydrogen_symbols.append(symbol)
                hydrogen_positions.append(pos[i])
            else:
                heavy_indices.append(i)
                heavy_symbols.append(symbol)
                heavy_positions.append(pos[i])
        
        # Match heavy atoms from SMILES order to actual heavy atoms
        # We need to match by type, handling duplicates
        ordered_heavy_indices = []
        heavy_atom_counts = {}
        for symbol in heavy_symbols:
            heavy_atom_counts[symbol] = heavy_atom_counts.get(symbol, 0) + 1
        
        smiles_atom_counts = {}
        for symbol in smiles_heavy_atoms:
            smiles_atom_counts[symbol] = smiles_atom_counts.get(symbol, 0) + 1
        
        # Match SMILES order to actual atoms
        used_indices = set()
        for smiles_symbol in smiles_heavy_atoms:
            # Find matching atom of this type that hasn't been used
            best_match = None
            best_distance = float('inf')
            
            for idx in heavy_indices:
                if idx in used_indices:
                    continue
                if atomic_symbols[idx] == smiles_symbol:
                    # If there are multiple of the same type, we could use distance
                    # For now, just take the first match
                    best_match = idx
                    break
            
            if best_match is not None:
                ordered_heavy_indices.append(best_match)
                used_indices.add(best_match)
        
        # Add any remaining heavy atoms that weren't matched (shouldn't happen, but safety)
        for idx in heavy_indices:
            if idx not in used_indices:
                ordered_heavy_indices.append(idx)
        
        # Combine: heavy atoms (in SMILES order) + hydrogens
        ordered_indices = ordered_heavy_indices + hydrogen_indices
        
        # Reorder
        ordered_symbols = [atomic_symbols[i] for i in ordered_indices]
        ordered_pos = pos[ordered_indices].tolist()
        
        return ordered_symbols, ordered_pos
            
    def _canonical_order_atoms(self, atomic_symbols, pos, smiles=None):
        return self._smiles_based_order_atoms(atomic_symbols, pos, smiles)

    def __len__(self):
        return len(self.qm9_dataset)

    def __getitem__(self, index):
        data = self.qm9_dataset[index]
        atomic_symbols = data['atomic_symbols']
        pos = data['pos']
        smiles = data.get('canonical_smiles') or data.get('smiles')  # Prefer canonical_smiles
        
        # Apply canonical ordering if requested
        if self.canonical_order:
            atomic_symbols, pos = self._canonical_order_atoms(atomic_symbols, pos, smiles)
        
        # Pad the atomic symbols and pos to the max length
        original_length = len(pos)
        pos = pos + [[0, 0, 0]] * (self.max_length - len(pos))
        pos = torch.tensor(pos, dtype=torch.float32)

        # Tokenize and pad the atomic symbols
        atomic_symbols = self.tokenizer.pad_tokenize(atomic_symbols, self.max_length)

        mask = torch.ones(self.max_length, dtype=torch.bool)
        mask[original_length:] = False
        return {'x': pos, 'y': atomic_symbols, 'mask': mask}

