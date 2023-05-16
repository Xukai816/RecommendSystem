import numpy as np
import torch
from torch import nn
from model.layers import CategoryLinear

'''
    因子分解机：Factorization Machines
'''


class FMCross(nn.Module):
    def __init__(self, field_nums, emd_dims):
        super().__init__()
        self.embeddings = nn.Embedding(sum(field_nums), emd_dims)
        torch.nn.init.xavier_uniform_(self.embeddings.weight.data)
        self.init_index = np.array((0, *np.cumsum(field_nums)[:-1]), dtype=np.compat.long)

    def forward(self, x):
        x = x + x.new_tensor(self.init_index).unsqueeze(0)
        emds = self.embeddings(x)
        square_of_sum = torch.sum(emds, dim=1) ** 2
        sum_of_square = torch.sum(emds ** 2, dim=1)
        ans = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)
        return ans


class FM(nn.Module):
    def __init__(self, field_nums, emd_dims):
        super().__init__()
        self.linear = CategoryLinear(field_nums, 1)
        self.cross = FMCross(field_nums, emd_dims)

    def forward(self, x):
        x = self.linear(x) + self.cross(x)
        x = x.squeeze(1)
        return torch.sigmoid(x.reshape(-1, 1))


class FM1(nn.Module):
    # 类别型特征的线性部分没有统一量纲,需要在输入时进行归一化
    # 该代码严格意义上不属于FM，因为特征向量是"特征域"的向量，而不是"特征"的向量
    def __init__(self, n=10, k=5):
        super().__init__()
        self.n = n  # 特征的数量
        self.k = k  # 向量的维度

        self.linear = nn.Linear(self.n, 1)  # 前两项线性层
        self.V = nn.Parameter(torch.randn(self.n, self.k))  # 交互矩阵
        nn.init.uniform_(self.V, -0.1, 0.1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # print(x.shape)
        # print(self.V.shape)
        # 线性部分
        linear_part = self.linear(x)
        # 交叉部分第一项：先相乘，再平方
        # #求出每一维度之和
        interaction_part_1 = torch.mm(x, self.V)
        interaction_part_1 = torch.pow(interaction_part_1, 2)
        # 交叉部分第二项：先平方，再相乘
        interaction_part_2 = torch.mm(torch.pow(x, 2), torch.pow(self.V, 2))
        output = linear_part.reshape((-1, 1)) + 0.5 * torch.sum(interaction_part_2 - interaction_part_1, dim=1,
                                                                keepdim=True)
        o1 = output.reshape(-1, 1)
        logit = self.sigmoid(o1)
        return logit

    # 不能对模型中不同类型的层个性化初始参数
    def initialize_parameters(self):
        for para in self.parameters():
            torch.nn.init.normal_(para)

    # 可针对不同的网络层，使用不同的参数初始化方法来初始化
    def initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight.data, 0, 0.01)
                layer.bias.data.zero_()


from torchstat import stat
from thop import profile

if __name__ == '__main__':
    model = FM([3, 4, 5, 1, 8], 10)
    from utils import params_count

    params_count(model)

    # input = torch.randn(32,10,5)
    # flops, params = profile(model, inputs=(input, ))
    # print('flops:{}'.format(flops))
    # print('params:{}'.format(params))
    # stat(model, (10, 5, 3))

    # model = FM(10, 5)
    # model.initialize_weights()
    # print("========")
    # x = torch.randn(1, 10)
    # output = model(x)
    # print(output)
