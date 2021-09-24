import os, sys, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import *
import pickle

class encoder(nn.Module):

    def __init__(self, nf, n_class, hidden_dim=128):
        super(encoder, self).__init__()
        self.n_class = n_class
        self.fc1 = nn.Linear(nf, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_class)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc3(x)

        return x

class DamDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.X[idx, ...], self.y[idx, ...]

if __name__ == "__main__":
    from sklearn.neural_network import MLPClassifier


    features, labels = pickle.load(open("./test_power_ratio_dam_data.pkl", "rb"))
    #features, labels = pickle.load(open("./test_other_ratio_dam_data.pkl", "rb"))
    features = np.array(features)
    labels = np.array(labels)

    scale = StandardScaler()
    features = scale.fit_transform(features)

    print ("features: ", features.shape, " ", "labels: ", labels.shape)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=199666)

    clf = MLPClassifier(random_state=19999, max_iter=500).fit(X_train, y_train)
    print (clf.score(X_test, y_test))

    exit(-1)

    train_loader = DataLoader(DamDataset(X_train, y_train), batch_size=128, shuffle=True)
    test_loader  = DataLoader(DamDataset(X_test, y_test), batch_size=32)

    model = encoder(features.shape[1], 4)
    #model = model.cuda()
    model = model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for ii in range(300):
        acc_train = 0
        trainloss = 0
        tot = 0
        tot_train = len(train_loader)
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.float()
            logit = model(batch_x)
            yhat = torch.argmax(logit, dim=1)
            acc_train += (yhat == batch_y).sum()
            CE_loss = torch.nn.CrossEntropyLoss()
            loss = CE_loss(logit, batch_y)
            trainloss += loss.item()
            tot += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print ("iteration %d: , acc: %.3f" % (ii, acc_train.item() / float(tot_train)))
        print ("avg train loss: ", trainloss / (tot))

        # Test
        model.eval()
        acc_test = 0
        tot_test = 0
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.float()
            with torch.no_grad():
                logit = model(batch_x)
                yhat = torch.argmax(logit, dim=1)
                print ("yhat: ", yhat, " ", batch_y)
                print ((yhat == batch_y).sum().item())
                acc_test += (yhat == batch_y).sum().item()
                tot_test += batch_x.size(0)
        print ("test acc: ", acc_test / float(tot_test))
        model.train()

print (features.shape)






