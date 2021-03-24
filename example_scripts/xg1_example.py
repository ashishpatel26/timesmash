#!/usr/bin/env python3
import pandas as pd
import sys
import numpy as np
from timesmash import Quantizer
from timesmash import XG1
from timesmash.utils import genesess
from EstimatorSelectionHelper import get_path

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


n_q = int(2000 / train_label.shape[0])  # limit number of quantizations if data is long
clast = XG1(n_quantizations=n_q)
a, b = clast.fit_transform(train=train, test=test, label=train_label)

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state=1)
clf.fit(a, train_label)
res = clf.predict(b)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

print("\nrandom forest_score:")
accur = accuracy_score(test_label, res)
print(accur)

from EstimatorSelectionHelper import EstimatorSelectionHelper

helper1 = EstimatorSelectionHelper()
helper1.fit(a, train_label)
help_predict = helper1.predict(b)
print(helper1.best_grid)


print("\nhelper_score:")
print(accuracy_score(test_label, help_predict))
