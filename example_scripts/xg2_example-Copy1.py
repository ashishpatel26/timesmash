#!/usr/bin/env python3
import pandas as pd
import sys
import numpy as np
from timesmash import Quantizer
from timesmash import XG2
from timesmash.utils import genesess
from EstimatorSelectionHelper import get_path, EstimatorSelectionHelper
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
sys.path.append('../')
from timesmash import _AUC_Feature



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

clast = XG2(clean=False)
a, b = clast.fit_transform(train=train, test=test, label=train_label)


# from EstimatorSelectionHelper import EstimatorSelectionHelper
# helper1 = EstimatorSelectionHelper()
# helper1.fit(a, train_label)
# help_predict = helper1.predict(b)
# print(helper1.best_grid)
# print("\nhelper_score:")
# print(accuracy_score(test_label, help_predict))

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state=1)
clf.fit(a, train_label)
# print(a)
res = clf.predict(b)


print("\nrandom forest_score:")
accur = accuracy_score(test_label, res)
print(1 - accur)
sys.exit()

from EstimatorSelectionHelper import EstimatorSelectionHelper

helper1 = EstimatorSelectionHelper()
helper1.fit(a, train_label)
help_predict = helper1.predict(b)
print(helper1.best_grid)


print("\nhelper_score:")
print(1 - accuracy_score(test_label, help_predict))
