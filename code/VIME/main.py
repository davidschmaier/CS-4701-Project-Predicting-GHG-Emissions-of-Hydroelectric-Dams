import os, sys, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import *
from self_vime import *
from sup_vime import *
from semi_vime import *
import pickle
from models import *

#if __name__ == "__main__":
def main_exp(**kwargs):
    features, labels = pickle.load(open("../test_power_ratio_dam_data.pkl", "rb"))
    unlab_features = pickle.load(open("../test_power_ratio_unsup_dam_data.pkl", "rb"))
    #features, labels = pickle.load(open("../test_other_ratio_dam_data.pkl", "rb"))
    #unlab_features = pickle.load(open("../test_other_ratio_unsup_dam_data.pkl", "rb"))
    features = np.array(features)
    labels = np.array(labels)

    scale = MinMaxScaler()
    #scale = StandardScaler()
    unlab_features = scale.fit_transform(unlab_features)
    features = scale.fit_transform(features)

    #unlab_features = np.concatenate((unlab_features, features), axis=0)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=199666)

    print ("features: ", features.shape, " ", "labels: ", labels.shape)
    print ("unlab features: ", unlab_features.shape)

    # Supervised from normal feature
    print ("start supervised training")
    sup_res = sup_train(X_train, y_train, X_test, y_test, nc=4)
    print ("sup res: ", sup_res)

    # Self supervised to get hidden dimension
    #encoder = vime_self(x_unlab=unlab_features, p_m=0.2, alpha=3, epochs=300, batch_size=128, verbose=False)
    print ("start self supervised training")
    encoder = vime_self(x_unlab=unlab_features, p_m=kwargs['p_m'], alpha=kwargs['alpha'], epochs=50, batch_size=kwargs['batch_size'], verbose=True)
    #encoder = Encoder(nf=unlab_features.shape[1])
    with torch.no_grad():
        hidden_rep_train = encoder(torch.tensor(X_train).float())
        hidden_rep_test  = encoder(torch.tensor(X_test ).float())
    hidden_rep_train = hidden_rep_train.cpu().data.numpy()
    hidden_rep_test  = hidden_rep_test.cpu().data.numpy()
    self_res = sup_train(hidden_rep_train, y_train, hidden_rep_test, y_test, nc=4)
    #semi_res = semi_train(X_train, y_train, unlab_features, X_test, y_test, encoder, p_m=0.2, K=3, beta=1.0) 
    print ("start semi supervised training")
    semi_res = semi_train(X_train, y_train, unlab_features, X_test, y_test, encoder, p_m=kwargs['p_m'], K=kwargs['K'], beta=kwargs['beta'], batch_size=kwargs['batch_size']) 
    print ("hidden size: ", hidden_rep_train.shape)
    print ("sup res: ", sup_res)
    print ("self res: ", self_res)
    print ("semi-sup res: ", semi_res)

    return sup_res, self_res, semi_res

if __name__ == "__main__":
    Ks = [3, 6, 8]
    #Ks = [3]
    p_ms = [0.1, 0.2, 0.3]
    #p_ms = [0.1]
    betas = [1.0, 2.0, 3.0]
    #betas = [1.0]
    alphas = [1.0, 2.0, 3.0]
    #alphas = [1.0]
    batch_sizes = [128, 256]
    #batch_sizes = [128]
    rr = 5

    best_sup_res = (-1, -1, -1)
    best_sup_parameters = None
    best_self_res = (-1, -1, -1)
    best_self_parameters = None
    best_semi_res = (-1, -1, -1)
    best_semi_parameters = None

    for K in Ks:
        for p_m in p_ms:
            for beta in betas:
                for alpha in alphas:
                    for batch_size in batch_sizes:
                        res = []
                        print ("current parameters: ", K, p_m, beta, alpha, batch_size)
                        for rep in range(rr):
                            res.append(main_exp(K=K, p_m=p_m, beta=beta, alpha=alpha, batch_size=batch_size))
                        sup_res = [x[0] for x in res]
                        self_res = [x[1] for x in res]
                        semi_res = [x[2] for x in res]
                        print ("Results!!!!: ", K, " ", p_m, " ", beta, " ", alpha, " ", batch_size)
                        print ("sup self semi: ", np.mean(sup_res), " ", np.mean(self_res), " ", np.mean(semi_res))
                        if np.mean(sup_res) > best_sup_res[0]:
                            best_sup_res = (np.mean(sup_res), np.mean(self_res), np.mean(semi_res), np.std(sup_res))
                            best_sup_parameters = [K, p_m, beta, alpha, batch_size]
                        if np.mean(self_res) > best_self_res[1]:
                            best_self_res = (np.mean(sup_res), np.mean(self_res), np.mean(semi_res), np.std(self_res))
                            best_self_parameters = [K, p_m, beta, alpha, batch_size]
                        if np.mean(semi_res) > best_semi_res[2]:
                            corr_sup_res = np.mean(sup_res)
                            corr_self_res = np.mean(self_res)
                            best_semi_res = (corr_sup_res, corr_self_res, np.mean(semi_res), np.std(semi_res))
                            best_semi_parameters = [K, p_m, beta, alpha, batch_size]
    print ("gs results:")
    print ("best sup: ")
    print (best_sup_res)
    print (best_sup_parameters)
    print ("best self: ")
    print (best_self_res)
    print (best_self_parameters)
    print ("best semi: ")
    print (best_semi_res)
    print (best_semi_parameters)





