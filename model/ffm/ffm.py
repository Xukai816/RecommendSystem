import torch
from torch import nn
import numpy as np
from model.layers import CategoryLinear


class FiledAwareFMCross(nn.Module):
    def __init__(self, field_dims, emd_dims):
        super().__init__()
        self.field_nums = len(field_dims)
        self.embeddings = nn.ModuleList([
            nn.Embedding(sum(field_dims), emd_dims) for _ in range(self.field_nums)
        ])
        self.init_index = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.compat.long)
        for embedding in self.embeddings:
            torch.nn.init.xavier_uniform_(embedding.weight.data)

    def forward(self, x):
        x = x + x.new_tensor(self.init_index).unsqueeze(0)  # unsqueeze(0)在外侧再套上一个括号
        cross_emb = [self.embeddings[i](x) for i in range(self.field_nums)]
        cross_val = list()
        for i in range(self.field_nums - 1):
            for j in range(i + 1, self.field_nums):
                cross_val.append(cross_emb[i][:, j] * cross_emb[j][:, i])
        ans = torch.stack(cross_val, dim=1)
        return ans


class FFM(torch.nn.Module):
    def __init__(self, field_dims, emd_dims):
        super().__init__()
        self.linear = CategoryLinear(field_dims, 1)
        self.ffm = FiledAwareFMCross(field_dims, emd_dims)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        ffm = torch.sum(torch.sum(self.ffm(x), dim=1), dim=1, keepdim=True)
        output = self.linear(x) + ffm
        o1 = output.reshape(-1, 1)
        ans = self.sigmoid(o1)
        return ans


if __name__ == '__main__':
    filed_dims = [6040, 3706, 2, 7, 21, 18]
    model = FFM(filed_dims, 10)
    inpt = torch.LongTensor([1, 2355, 1, 4, 6, 10])
    oupt = model.forward(inpt)
    print(oupt)

    # t1 = []
    # for i in range(10):
    #     t1.append(torch.tensor([i]))
    # b = torch.stack(t1, dim=1)
    # c = torch.stack(t1, dim=0)
    # print(t1)
    # print(b)
    # print(c)
