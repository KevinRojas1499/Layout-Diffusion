import torch

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

