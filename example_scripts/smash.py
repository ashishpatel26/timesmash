#!/usr/bin/env python3
import pandas as pd
import sys
import numpy as np
sys.path.append('../')
import pickle
from timesmash import InferredHMMLikelihood as Smash_featurizer
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

clast = Smash_featurizer(n_quantizations=10)

with open('pickleddata', 'wb') as f:
    pickle.dump(clast, f)
with open('pickleddata', 'rb') as f:
    clast = pickle.load(f)
      
a, b = clast.fit_transform(train=train, test=test, label=train_label)

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state=1)
clf.fit(a, train_label)
res = clf.predict(b)

print("\nrandom forest_score:")
accur = accuracy_score(test_label, res)
print(accur)
