import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

'''
    一次性将所有数据读进内存
'''


class MovieLens(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.df = pd.read_csv(data_path)
        # 增加一个数值类型的用户特征：活跃度
        # self.df['vatility'] = list(np.random.random(self.df.shape[0]))
        self.df = self.df[['UserID', 'MovieID', 'gender', 'age', 'occupation', 'GenresLe', 'Rating']]
        self.label = [1 if c > 2 else 0 for c in self.df['Rating']]
        self.df = self.df.drop('Rating', axis=1).values

    def __getitem__(self, index):
        line = torch.tensor(self.df[index], dtype=torch.int)  # .view(1, -1)
        labels = torch.tensor(self.label[index], dtype=torch.float)
        return line, labels

    def __len__(self):
        return self.df.shape[0]


if __name__ == '__main__':
    dataset = MovieLens('./data/ml-1m/rating_user_movie_merge.csv')
    fn, label = dataset[10]
    print(fn)
    print('='*10)
    print(label)
