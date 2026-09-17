import torch
import torch.nn as nn


class GaussianSampler(nn.Module):
    '''
    Samples the flow-matching prior x0 ~ N(0, I) per element, and maps ground-truth
    data (assumed normalized to [0, 1]) into the same [-1, 1] range the prior lives in.
    Upstream LayoutFlow supports several other priors (uniform, mixtures, a "frame"
    prior); only the gaussian one is used by its default config, so that's all this
    keeps. Swap this module out to experiment with a different prior.
    '''

    def __init__(self, out_dim=9):
        super().__init__()
        self.out_dim = out_dim

    def sample(self, batch):
        B, S = batch['type'].shape
        device = batch['type'].device
        x0 = torch.zeros((B, S, self.out_dim), device=device)
        for i in range(B):
            L = batch['length'][i]
            x0[i, :L] = torch.randn((L, self.out_dim), device=device)
        return x0

    def preprocess(self, data, reverse=False):
        return (data + 1) / 2 if reverse else 2 * data - 1
