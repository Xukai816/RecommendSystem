import torch
import random
import pandas as pd


class GenDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str):

        with open(data_path, "r") as f:
            self.data = f.readlines()
            # 如果这里都爆内存的话，
            # 看起来只能使用文件指针，在getitem里边逐行读取了
            # 得到的data是 list[str]
        random.shuffle(self.data)
        self.data_gen = self.get_data()

    def get_data(self):

        for doc in self.data:
            # 每个doc是一行文本，可能因为过长处理成为多个samples
            batch_samples = []
            # 巴拉巴拉
            # 经过处理得到了batch_samples

            while len(batch_samples) > 0:
                # 逐个把数据返回,每次只返回一条
                yield batch_samples.pop()

    def __len__(self):
        # 这里返回长度是用于tqdm进度条显示用的
        # 我这里乘以4是我之前预处理的时候看得到总量大概是文档数目的4倍
        # 你也可以设定一个很大的数字，当dataloader提取不到数据的时候就会停止
        return len(self.data * 4)

    def __getitem__(self, idx):
        # 每次使用next函数返回生成器生成的一条数据，此处的idx用不到了
        return next(self.data_gen)
