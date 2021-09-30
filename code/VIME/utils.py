import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

def mask_generator(p, x):
    return np.random.binomial(1, p, x.shape)

def corrupted_generator(m, x):
    no, dim = x.shape
    x_bar = np.zeros([no, dim])
    for i in range(dim):
        idx = np.random.permutation(no)
        x_bar[:, i] = x[idx, i]
    
    x_tilde = x * (1 - m) + x_bar * m
    m_new = 1 * (x != x_tilde)

    return m_new, x_tilde

class DamDataset(Dataset):

    def __init__(self, X, y=None):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.y is None:
            return self.X[idx, ...]
        else:
            return self.X[idx, ...], self.y[idx, ...]


