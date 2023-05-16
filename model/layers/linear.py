import torch
import numpy as np


class CategoryLinear(torch.nn.Module):
    def __init__(self, field_dims, emd_dims=1):
        super().__init__()
        self.embeddings = torch.nn.Embedding(sum(field_dims), emd_dims)
        self.bias = torch.nn.Parameter(torch.zeros(emd_dims))
        self.emd_index_start = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.compat.long)

    def forward(self, x):
        x = x + x.new_tensor(self.emd_index_start).unsqueeze(0)
        return torch.sum(self.embeddings(x), dim=1) + self.bias
