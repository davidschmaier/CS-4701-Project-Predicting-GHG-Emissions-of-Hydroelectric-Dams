import numpy as np
from utils import *
from models import *
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def evaluate(loader, model, encoder):
    model.eval()
    acc = 0
    acc_tot = 0.
    tot = 0.
    loss = 0.
    with torch.no_grad():
        for X, y in loader:
            X = encoder(X.float())
            yhat = model(X)
            acc_tot += X.size(0)
            acc += (torch.argmax(yhat, dim=1) == y).sum().item()
            tot += 1
            CE_loss = nn.CrossEntropyLoss()
            loss += CE_loss(yhat, y).item()
    print ("evaluation acc: ", (acc / acc_tot))
    print ("evaluation loss: ", loss / tot)
    model.train()

    return float(acc / acc_tot), loss / tot

def semi_train(X_train, y_train, Xu_train, X_test, y_test, encoder, p_m, K, beta, batch_size):

    tot = X_train.shape[0]
    val_idx = int(tot * 0.9)
    X_val, y_val = X_train[val_idx:], y_train[val_idx:]
    X_train, y_train = X_train[:val_idx], y_train[:val_idx]

    ftensor = lambda x : torch.tensor(x).float()
    ltensor = lambda x : torch.tensor(x).long()
    X_train, y_train = ftensor(X_train), ltensor(y_train)
    X_val, y_val = ftensor(X_val), ltensor(y_val)
    X_test, y_test = ftensor(X_test), ltensor(y_test)
    Xu_train = ftensor(Xu_train)

    model = sup_model(nf=128, nc=4)
    encoder.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()

    train_loader = DataLoader(DamDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(DamDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(DamDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    best_val_loss = 1e10
    best_epoch = -1
    best_model = None
    for ii in range(1000):
        ce_loss = 0.
        con_loss = 0.
        tot = 0.
        for X, y in train_loader:
            tot += 1
            # Encoder labeled data
            with torch.no_grad():
                X = encoder(X)
            # Select unlabeled data
            batch_u_idx = np.random.permutation(Xu_train.size(0))[:batch_size]
            Xu_ori = Xu_train[batch_u_idx, ...]

            Xu = []

            # Augement unlabeled data
            for rep in range(K):
                m_batch = mask_generator(p_m, Xu_ori.cpu().data.numpy())
                _, Xu_temp = corrupted_generator(m_batch, Xu_ori.cpu().data.numpy())
                Xu_temp = ftensor(Xu_temp)
                
                with torch.no_grad():
                    Xu_temp = encoder(Xu_temp)
                Xu.append(Xu_temp)
            Xu = torch.stack(Xu)
            y_hat_logit = model(X)
            yu_hat_logit = model(Xu)
            #std = yu_hat_logit.std(0)
            var = torch.var(yu_hat_logit, dim=0, unbiased=False)
            #print (yu_hat_logit[:, 2, 2])
            #print (yu_hat_logit.mean(0)[2, 2])
            #print (var[2, 2])
            #exit(-1)

            CE_loss = nn.CrossEntropyLoss()
            loss1 = CE_loss(y_hat_logit, y)
            ce_loss += loss1.item()
            loss2 = torch.mean(var)
            con_loss += loss2.item()

            loss = loss1 + beta * loss2

            if torch.isnan(loss):
                print ("nan!!!!!!!!!!!!!!!!!!!")
                return None

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print ("ce loss: ", ce_loss / tot)
        print ("consist loss: ", con_loss / tot)
        print ("validation on epoch: %d" % (ii))
        val_acc, val_loss = evaluate(val_loader, model, encoder)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = ii
            best_model = copy.deepcopy(model)
        else:
            if ii - best_epoch >= 100:
                print ("early stopping")
                break
    print ("Test resuts: ")
    test_res, test_loss = evaluate(test_loader, best_model, encoder)
    return test_res







