import torch
import pandas as pd
import numpy as np


if __name__ == '__main__':
    # emd = torch.nn.Embedding(4,5)
    # print('emd mm')
    # print(emd.weight)
    # print('word1')
    # word1 = torch.LongTensor(1)
    # emd_1 = emd[word1]
    # print(emd_1)

    word1 = torch.LongTensor([0, 1, 2])
    word2 = torch.LongTensor([3, 1, 2])
    embedding = torch.nn.Embedding(4, 5)
    print(embedding.weight)
    print('word1:')
    print(embedding(word1))
    print('word2:')
    print(embedding(word2))
    print('ooo')
    print(embedding(torch.LongTensor([0])))