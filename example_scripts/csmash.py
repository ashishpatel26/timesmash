#!/usr/bin/env python3
def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn


import pandas as pd
import sys
from sklearn.linear_model import SGDClassifier
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from timesmash import Quantizer
from timesmash import Smash as Smash_featurizer
from timesmash.utils import genesess
from sklearn.cluster import (
    KMeans,
    AffinityPropagation,
    MeanShift,
    SpectralClustering,
    AffinityPropagation,
    DBSCAN,
    Birch,
    AgglomerativeClustering,
)
from sklearn.svm import SVC
from EstimatorSelectionHelper import EstimatorSelectionHelper, get_path
from sklearn.metrics import accuracy_score
import json

percent_run = 0.2
np.random.seed(4)
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

_qtz = Quantizer(n_quantizations=1)
_qtz.fit(train, labels=train_label)
best_score = 0
best_grid = None
best_clast = None
best_b = None
best_n_expend = 0
for n_expend in [2]:
    # clust = [AffinityPropagation(), KMeans(n_clusters = n_expend, random_state=0), MeanShift(), SpectralClustering(assign_labels="discretize", affinity='precomputed', n_clusters = n_expend, random_state=0), SpectralClustering(n_clusters = n_expend, random_state=0, affinity='precomputed'), AgglomerativeClustering(n_clusters = n_expend), AffinityPropagation(affinity='precomputed'), Birch(n_clusters = n_expend)]
    clust = [KMeans(n_clusters=n_expend, random_state=0)]
    for clas in clust:
        y_train = train_label.copy()
        X_train = train.copy()
        for label, dataframe in y_train.groupby(y_train.columns[0]):
            if dataframe.index.shape[0] < 3:
                continue
            clast = Lsmash_distance(quantizer=_qtz)
            clast.fit(X_train.loc[dataframe.index])
            dist = clast.produce()
            kmeans = clas.fit(dist)
            clasters = kmeans.labels_
            new_l = [str(label) + "_" + str(i) for i in clasters]
            y_train.loc[dataframe.index, y_train.columns[0]] = new_l
        clast = Smash_featurizer(quantizer=_qtz)
        a, b = clast.fit_transform(train=X_train, test=test, label=y_train)
        a.fillna(0, inplace=True)
        b.fillna(0, inplace=True)
        """
        clf = RandomForestClassifier(random_state=1)
        clf.fit(a,train_label)

        pd.to_pickle(a, "./train_csmash{}.pkl".format(dataset))
        pd.to_pickle(b, "./test_csmash{}.pkl".format(dataset))
        helper1 = EstimatorSelectionHelper()
        helper1.fit(a, train_label)
        best_score_ = helper1.best_grid.best_score_
        if best_score_> best_score:
            best_score = best_score_                     
            best_grid = helper1.best_grid
            best_clast = clas
            best_b = b
            best_n_expend = n_expend        
            helper_predict = best_grid.predict(best_b)
            print('n clasters:'+str(best_n_expend))
            print(best_clast)
            print(helper1.best_grid)
            accur = accuracy_score(test_label, helper_predict)
            print("\nhelper_score so far :")
            print(1-accur)
         
helper_predict = best_grid.predict(best_b)
print('n clasters:'+str(best_n_expend))
print(best_clast)
print(helper1.best_grid)
accur = accuracy_score(test_label, helper_predict)
print("\n max helper_score:")
print(1-accur)
 """
clf = RandomForestClassifier(random_state=1)
clf.fit(a, train_label)
res = clf.predict(b)
accur = accuracy_score(test_label, res)
print(1 - accur)
