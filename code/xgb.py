import os, sys, copy
import numpy as np
import xgboost as xgb
import pickle

#features, labels = pickle.load(open("./test_power_dam_data.pkl", "rb"))
#features, labels = pickle.load(open("./test_other_dam_data.pkl", "rb"))
#features, labels = pickle.load(open("./test_power_ratio_dam_data.pkl", "rb"))
features, labels = pickle.load(open("./test_other_ratio_dam_data.pkl", "rb"))

features = np.array(features)
labels = np.array(labels)

print (features.shape)
print (labels.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=199666)

d_train = xgb.DMatrix(X_train, label=y_train)
d_test  = xgb.DMatrix(X_test,  label=y_test)

param = {'max_depth': 10, 'num_class': 2, 'eta': 1, 'objective': 'multi:softprob'}
num_round = 10

bst = xgb.train(param, d_train, num_round)

preds = bst.predict(d_test)
preds = np.array(preds)

tot = y_test.shape[0]
acc = 0

for i in range(tot):
    print (preds[i])
    if np.argmax(preds[i]) == y_test[i]:
        acc += 1

print ("acc: ", 1.0 * acc / tot)

