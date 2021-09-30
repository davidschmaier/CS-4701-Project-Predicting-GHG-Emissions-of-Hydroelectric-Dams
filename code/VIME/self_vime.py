import numpy as np
from utils import *
from models import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def vime_self(x_unlab, p_m, alpha, **parameters):
    _, dim = x_unlab.shape
    epochs = parameters['epochs']
    batch_size = parameters['batch_size']

    encoder = Encoder(nf=dim)
    model = self_model(nf=dim)

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(model.parameters()), lr=0.001)

    data_loader = DataLoader(DamDataset(x_unlab), batch_size=batch_size, shuffle=True)

    for ii in range(epochs):
        ce_loss = 0.
        mse_loss = 0.
        acc = 0.
        acc_tot = 0.
        tot = 0.
        for X in data_loader:
            tot += 1
            acc_tot += X.size(0)
            X = X.float()
            m_unlab = mask_generator(p_m, X.cpu().data.numpy())
            m_label, x_tilde = corrupted_generator(m_unlab, X.cpu().data.numpy())
            m_label = torch.tensor(m_label).float()
            x_tilde = torch.tensor(x_tilde).float()

            h = encoder(X)
            h = h.float()
            mask_hat, feature_hat = model(h)

            mask_hat = mask_hat.float()
            feature_hat = feature_hat.float()

            CE_loss = nn.BCELoss()
            MSE_loss = nn.MSELoss()

            #print (mask_hat.cpu().data.numpy())
            #print (m_label.cpu().data.numpy())
            if False and parameters['verbose']:
                print ("predict: ", feature_hat.cpu().data.numpy()[0])
                print ("corrupted: ", x_tilde.cpu().data.numpy()[0])
                print ("original: ", X.cpu().data.numpy()[0])
                print ("predcit mask: ", mask_hat.cpu().data.numpy()[0])
                print ("mask label: ", m_label.cpu().data.numpy()[0])
            
            loss = CE_loss(mask_hat, m_label) + alpha * MSE_loss(feature_hat, X)
            #loss = CE_loss(mask_hat, m_label)
            ce_loss += CE_loss(mask_hat, m_label).item()
            mse_loss += MSE_loss(feature_hat, X).item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if parameters['verbose']:
            print ("Iteration: %d, CE loss: %.3f, mse_loss: %.3f" % (ii, ce_loss / float(tot), mse_loss / float(tot)))
            
    return encoder
            
    

    

