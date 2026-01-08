import torch
import re
from torch import Tensor

class IntervalTokenizer():
    
    def __init__(self, left_endpoint: int, right_endpoint: int, num_bins: int) -> None:
        self.left_endpoint = left_endpoint
        self.right_endpoint = right_endpoint
        self.num_bins = num_bins
        self.bins = torch.linspace(left_endpoint, right_endpoint, num_bins)
        self.bin_width = (right_endpoint - left_endpoint) / num_bins
        
    def tokenize(self, x: float) -> int:
        return torch.bucketize(x, self.bins)
    
    def decode(self, tokens: int) -> float:
        return self.left_endpoint + (tokens + torch.rand_like(tokens)) * self.bin_width

class CharacterTokenizer():
    def __init__(self, characters: str):
        self.characters = characters
        self.char_to_idx = {char: idx for idx, char in enumerate(characters)}
        self.idx_to_char = {idx: char for idx, char in enumerate(characters)}
        self.char_to_idx['<pad>'] = len(characters)
        self.char_to_idx['<M>'] = len(characters) + 1
        self.idx_to_char[len(characters)] = '<pad>'
        self.idx_to_char[len(characters) + 1] = '<M>'
        self.vocab_size = len(characters)
        self.pad_token = '<pad>'
        self.mask_token = '<M>'
        self.eos_token = '<EOS>'
        self.bos_token = '<BOS>'

    def tokenize(self, x: str) -> int:
        return torch.tensor([self.char_to_idx[char] for char in x], dtype=torch.long)

    def decode(self, tokens: Tensor) -> str:
        return ''.join([self.idx_to_char[token.item()] for token in tokens])



class VocabTokenizer():
    def __init__(self, vocab: set):
        self.vocab = vocab
        self.vocab_size = len(self.vocab)
        self.atom_to_idx = {atom: idx for idx, atom in enumerate(self.vocab)}
        self.idx_to_atom = {idx: atom for atom, idx in self.atom_to_idx.items()}
        # Pattern to match special tokens in <token> format
        self.special_token_pattern = re.compile(r'<[^>]+>')
        self.pad_token = '<pad>'
        self.mask_token = '<M>'
        self.eos_token = '<EOS>'
        self.bos_token = '<BOS>'
        self.pad_token_id = self.add_token(self.pad_token)
        self.mask_token_id = self.add_token(self.mask_token)
        self.eos_token_id = self.add_token(self.eos_token)
        self.bos_token_id = self.add_token(self.bos_token)

    def tokenize(self, x: str) -> Tensor:
        """
        Tokenize a string, handling special tokens in <token> format.
        Special tokens are treated as single tokens, while other characters
        are tokenized individually.
        """
        tokens = []
        i = 0
        while i < len(x):
            # Check if we're at the start of a special token
            if x[i] == '<':
                # Find the closing '>'
                end = x.find('>', i)
                if end != -1:
                    # Extract the special token including angle brackets
                    special_token = x[i:end+1]
                    if special_token in self.atom_to_idx:
                        tokens.append(self.atom_to_idx[special_token])
                    else:
                        raise ValueError(f"Unknown special token: {special_token}")
                    i = end + 1
                else:
                    # No closing '>', treat '<' as regular character
                    if x[i] in self.atom_to_idx:
                        tokens.append(self.atom_to_idx[x[i]])
                    else:
                        raise ValueError(f"Unknown token: {x[i]}")
                    i += 1
            else:
                # Regular character tokenization
                if x[i] in self.atom_to_idx:
                    tokens.append(self.atom_to_idx[x[i]])
                else:
                    raise ValueError(f"Unknown token: {x[i]}")
                i += 1
        
        return torch.tensor(tokens, dtype=torch.long)

    def decode(self, tokens: Tensor) -> str:
        return ''.join([self.idx_to_atom[token.item()] for token in tokens])
    
    def add_token(self, token: str):
        """
        Add a token to the vocabulary. If the token is in <token> format,
        it will be added as a special token.
        """
        # Validate special token format if it starts with '<'
        if token.startswith('<'):
            if not token.endswith('>'):
                raise ValueError(f"Special token must end with '>': {token}")
            if not self.special_token_pattern.match(token):
                raise ValueError(f"Invalid special token format: {token}")
        
        if token in self.vocab:
            raise ValueError(f"Token already exists in vocabulary: {token}")
        
        self.vocab.add(token)
        self.vocab_size = len(self.vocab)
        self.atom_to_idx[token] = self.vocab_size - 1
        self.idx_to_atom[self.vocab_size - 1] = token
        return self.vocab_size - 1
    
    def is_special_token(self, token: str) -> bool:
        """Check if a token is in the special <token> format."""
        return bool(self.special_token_pattern.match(token))
    
    def pad_tokenize(self, x: str, max_length: int) -> Tensor:
        return torch.cat([self.tokenize(x), torch.full((max_length - len(x),), self.pad_token_id, dtype=torch.long)])