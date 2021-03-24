#!/usr/bin/env python3
import pandas as pd
import sys
import numpy as np
from timesmash import Quantizer
from timesmash import Smash_featurizer, Smash_featurizer_state
from timesmash.utils import genesess
from sklearn import tree
from sklearn.svm import SVC
from EstimatorSelectionHelper import get_path
from sklearn.metrics import accuracy_score

np.random.seed()
import random

random.seed()

percent_run = 0.2

# https://archive.ics.uci.edu/ml/datasets/Synthetic+Control+Chart+Time+Series
dataset = sys.argv[1]
train_path, test_path = get_path(dataset)
print(train_path)
train = pd.read_csv(train_path, header=None, delim_whitespace=True, na_values="NaN")
test = pd.read_csv(test_path, header=None, delim_whitespace=True, na_values="NaN")

train_label = pd.DataFrame(train[0].copy())
del train[0]
test_label = pd.DataFrame(test[0].copy())
del test[0]
from sklearn.model_selection import train_test_split

clast = Smash_featurizer(eps=0.2, n_quantizations=1)
a, b = clast.fit_transform(train=train, test=test, label=train_label)
"""X_test,X_train,y_test,y_train= train_test_split(train, labels, test_size=1-percent_run, random_state = 1) 

X_train = X_train.sort_index()
y_train = y_train.sort_index()
X_test = X_test.sort_index()
y_test = y_test.sort_index()

from EstimatorSelectionHelper import EstimatorSelectionHelper
best_score = 0
best_grid = None
best_eps = 0
best_b = None
for ep in [0.2, 0.1, 0.01]:
    clast = Smash_featurizer(eps = ep, n_quantizations=100)
    a,b=clast.fit_transform(train=train, test=test, label=train_label)
    helper1 = EstimatorSelectionHelper()
    helper1.fit(a, train_label)
    best_score_ = helper1.best
    print(best_score_)
    if best_score_>= best_score:
        best_score = best_score_                     
        best_grid = helper1.best_grid
        best_eps = ep
        print("\n")
        print("eps: " + str(best_eps))
        print("\n")                            
        print(best_grid)
        best_b = b
        help_predict = best_grid.predict(b)
        print("\nhelper_score so far :")
        print(accuracy_score(test_label, help_predict))

print("\n")
print("eps: " + str(best_eps))
print("\n")                            
print(best_grid)
help_predict = best_grid.predict(best_b)
print("\nbest helper_score:")
print(accuracy_score(test_label, help_predict))
"""
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state=1)
clf.fit(a, train_label)
res = clf.predict(b)

print("\nrandom forest_score:")
accur = accuracy_score(test_label, res)
print(accur)
