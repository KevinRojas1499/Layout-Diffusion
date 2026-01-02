import torch
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
