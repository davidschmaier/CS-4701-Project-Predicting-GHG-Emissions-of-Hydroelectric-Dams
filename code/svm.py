from sklearn import svm
import numpy as np
import os, sys, copy
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import *

features, labels = pickle.load(open("./test_power_ratio_dam_data.pkl", "rb"))
#features, labels = pickle.load(open("./test_other_ratio_dam_data.pkl", "rb"))
features = np.array(features)
labels = np.array(labels)

print (features[:10, 3])
#features = scale(features)
scale = StandardScaler()
features = scale.fit_transform(features)
print (features[:10, 3])

print ("features: ", features.shape, " ", "labels: ", labels.shape)

for i in range(labels.shape[0]):
    labels[i] += 1
    if labels[i] > 2:
        labels[i] -= 5

print (labels[:20])


X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=199666)
X_train_cp = copy.deepcopy(X_train)
y_train_cp = copy.deepcopy(y_train)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=199666666)

Cs = [1.0 / (1 << 5), 1.0 / (1 << 3), 1.0 / (1 << 1), 2, (1 << 3), (1 << 5), (1 << 7), (1 << 9), (1 << 11), (1 << 13), (1 << 15)]
gammas = [1.0 / (1 << 15), 1.0 / (1 << 13), 1.0 / (1 << 11), 1.0 / (1 << 9), 1.0 / (1 << 7), 1.0 / (1 << 5), 1.0 / (1 << 3), 1.0 / (1 << 1), (1 << 1), (1 << 3)]

best_val_acc = -1
best_parameters = None
for C in Cs:
    for gamma in gammas:
        classifier = svm.SVC(verbose=False, C=C, gamma=gamma)
        classifier.fit(X_train, y_train)
        val_res = classifier.predict(X_val)
        num_correct = (val_res == y_val).sum()
        val_acc = num_correct / len(y_val)
        print ("val acc: ", val_acc, " ", "best val acc: ", best_val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_parameters = [C, gamma]

classifier = svm.SVC(verbose=False, C=best_parameters[0], gamma=best_parameters[1])
classifier.fit(X_train_cp, y_train_cp)
test_res = classifier.predict(X_test)
num_correct = (test_res == y_test).sum()
test_acc = num_correct / len(y_test)
print ("config: ", best_parameters)
print ("test acc: ", test_acc)
"""
classifier = svm.SVC(verbose=True)
classifier.fit(X_train, y_train)
results = classifier.predict(X_train)
num_correct = (results == y_train).sum()
acc = num_correct / len(y_train)
print ("train: ", acc)

results = classifier.predict(X_test)
num_correct = (results == y_test).sum()
acc = num_correct / len(y_test)
print ("test: " , acc)
"""
