import os, sys, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    
    def __init__(self, nf, hidden_dim=128):
        super(Encoder, self).__init__()
        self.h = nn.Linear(nf, hidden_dim)
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        h = F.relu(self.h(x))
        h = F.relu(self.fc(h))

        return h

class self_model(nn.Module):

    def __init__(self, nf, hidden_dim=128):
        super(self_model, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, nf)
        self.fc2 = nn.Linear(hidden_dim, nf)

    def forward(self, h):
        mask_predict = F.sigmoid(self.fc1(h))
        feature_estimate = F.sigmoid(self.fc2(h))

        return mask_predict, feature_estimate

class sup_model(nn.Module):

    def __init__(self, nf, nc, hidden_dim=128):
        super(sup_model, self).__init__()
        self.nc = nc
        self.nf = nf
        self.h = nn.Linear(nf, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, nc)

    def forward(self, x):
        h = F.relu(self.h(x))
        h = F.relu(self.fc1(h))
        logit = self.fc2(h)

        return logit
        
