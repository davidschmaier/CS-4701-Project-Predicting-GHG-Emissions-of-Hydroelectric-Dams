import os, sys, copy
import numpy as np
import xgboost as xgb
import pickle
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#features, labels = pickle.load(open("./test_power_dam_data.pkl", "rb"))
#features, labels = pickle.load(open("./test_other_dam_data.pkl", "rb"))
features, labels = pickle.load(open("./test_power_ratio_dam_data.pkl", "rb"))
#features, labels = pickle.load(open("./test_other_ratio_dam_data.pkl", "rb"))

features = np.array(features)
labels = np.array(labels)

print (features.shape)
print (labels.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=199666)
X_train_cp = copy.deepcopy(X_train)
y_train_cp = copy.deepcopy(y_train)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=199666666)

#param = {'max_depth': 10, 'num_class': 8, 'eta': 1, 'objective': 'multi:softprob'}
#param = {'n_estimators': 1000, 'verbose': 2, 'max_depth': 8, 'learning_rate': 0.001, 'num_class': 4, 'eta': 0.3, 'objective': 'multi:softprob'}
model = xgb.XGBClassifier(max_depth=8, learning_rate=0.05, n_estimators=100, silent=True, objective="multi:softprob")

evalset = [(X_train, y_train), (X_val, y_val)]
#model.fit(X_train, y_train, eval_set=evalset, eval_metric="logloss")
best_val_acc = -1
best_feature = None

lrs = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11]
max_depths = [x for x in range(3, 12)]
n_estimators = [x for x in range(100, 800, 100)]
etas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for lr in lrs:
    for md in max_depths:
        for ne in n_estimators:
            for eta in etas:
                model = xgb.XGBClassifier(max_depth=md, learning_Rate=lr, n_estimators=ne, silent=True, objective="multi:softprob", eta=eta)
                model.fit(X_train, y_train, eval_set=evalset)

                yhat = model.predict(X_val)
                valid_acc = accuracy_score(y_val, yhat)
                print ("val acc: ", accuracy_score(y_val, yhat), "best acc: ", best_val_acc)
                if valid_acc > best_val_acc:
                    best_val_acc =valid_acc 
                    best_feature = [lr, md, ne, eta]

print ("Best test acc: ", best_val_acc)
print ("best feature: ", best_feature)

model = xgb.XGBClassifier(max_depth=best_feature[1], learning_Rate=best_feature[0], n_estimators=best_feature[2], eta=best_feature[3], silent=True, objective="multi:softprob")
model.fit(X_train_cp, y_train_cp)

yhat = model.predict(X_test)
test_acc = accuracy_score(yhat, y_test)
print ("Test acc: ", test_acc)

