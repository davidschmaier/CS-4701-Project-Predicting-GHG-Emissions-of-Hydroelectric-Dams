import os, sys, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import *
from sklearn.neural_network import MLPClassifier
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from models import sup_model
from torch.utils.data import Dataset, DataLoader
from utils import *

################# Scipy interface #############################
#def sup_train(X_train, y_train, X_test, y_test):
#    parameter_space = {
#                'hidden_layer_sizes': [(32), (64), (128), (256)],
#                'activation': ['relu'],
#                'solver': ['sgd', 'adam'],
#                'alpha': [0.0001, 0.05],
#                'learning_rate': ['constant','adaptive'],
#            }
#    #mlp = MLPClassifier(hidden_layer_sizes=(64), random_state=19999, max_iter=300, early_stopping=True)
#    mlp = MLPClassifier(random_state=19999, max_iter=300, early_stopping=True)
#    clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3) 
#    clf.fit(X_train, y_train)
#    return clf.score(X_test, y_test)
###############################################################

def evaluate(loader, model):
    model.eval()
    acc = 0
    acc_tot = 0.
    tot = 0.
    loss = 0.
    with torch.no_grad():
        for X, y in loader:
            X = X.float()
            yhat = model(X)
            acc_tot += X.size(0)
            acc += (torch.argmax(yhat, dim=1) == y).sum().item()
            tot += 1
            CE_loss = nn.CrossEntropyLoss()
            loss += CE_loss(yhat, y).item()
    print ("evaluation results: ", (acc / acc_tot))
    model.train()

    return float(acc / acc_tot), loss / tot

def sup_train(X_train, y_train, X_test, y_test, nc):    

    model = sup_model(nf=X_train.shape[1], nc=nc)

    tot = X_train.shape[0]
    val_idx = int(tot * 0.9)
    X_val, y_val = X_train[val_idx:], y_train[val_idx:]
    X_train, y_train = X_train[:val_idx], y_train[:val_idx]
    ftensor = lambda x : torch.tensor(x).float()
    ltensor = lambda x : torch.tensor(x).long()
    X_train, y_train = ftensor(X_train), ltensor(y_train)
    X_val, y_val = ftensor(X_val), ltensor(y_val)
    X_test, y_test = ftensor(X_test), ltensor(y_test)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
    model.train()

    train_loader = DataLoader(DamDataset(X_train, y_train), batch_size=128, shuffle=True)
    val_loader   = DataLoader(DamDataset(X_val, y_val), batch_size=128, shuffle=False)
    test_loader  = DataLoader(DamDataset(X_test, y_test), batch_size=128, shuffle=False)
    best_val_loss = 1e10
    best_epoch = -1
    best_model = None
    for ii in range(300):
        ce_loss = 0.
        tot = 0.
        for X, y in train_loader:
            tot += 1
            X = X.float()

            yhat = model(X)
            CE_loss = nn.CrossEntropyLoss()
            loss = CE_loss(yhat, y)
            ce_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print ("validatin on epoch: %d" % (ii))
        val_acc, val_loss = evaluate(val_loader, model)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = ii
            best_model = copy.deepcopy(model)
        else:
            if ii - best_epoch >=  100:
                print ("early stopping")
                break

    print ("test results: ")
    test_res, test_loss = evaluate(test_loader, best_model)
    return test_res





