import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, X, target):
        self.X = X
        self.target = target
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        target = np.array([self.target[index]])
        sample = {'samples': torch.from_numpy(self.X[index]).float(),
                  'targets': torch.from_numpy(target)}
        return sample
    
    @staticmethod
    def load_data(path, test=False):
        data = pd.read_csv(path, index_col='Id')
        features = data.columns[:-1].values.tolist()
        
        if test:
            X, features = data[data.columns].values, data.columns
            return X, features
        
        df = data[features].apply(lambda x: np.sum(x), axis=1)
        data = data[df > 0]
        data = data[np.all(data >= 0, axis=1)]
        
        X, y = data[data.columns[:-1]].values, data[data.columns[-1]].values
        
        print(f'Total samples: {len(data)}')
        counts = np.unique(data["Category"].values, return_counts=True)[1]
        print(f'Samples per class\n\tmean: {np.mean(counts):.0f}\n\tmedian: {np.median(counts):.0f}')
        
        return X, y, features