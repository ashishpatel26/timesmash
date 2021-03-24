import pandas as pd
import pickle
import sys
import numpy as np
from timesmash import Quantizer, XHMMFeatures, InferredHMMLikelihood
from sklearn.cluster import KMeans
import pickle
import os
from sklearn.cluster import KMeans

def is_equal_set_of_sets(ss1,ss2):
    if len(ss1) != len(ss2):
        return False
    for s1 in ss1:
        found = False
        for s2 in ss2:
            if s1==s2:
                found = True
                break
        if not found:
            return False
    return True

def is_equal_labels(l1,l2):
    ss1 = []
    for lb, dataframe in l1.groupby(l1.columns[0]):
        ss1.append(frozenset(dataframe.index))
    ss2 = []
    for lb, dataframe in l2.groupby(l2.columns[0]):
        ss2.append(frozenset(dataframe.index))
    return is_equal_set_of_sets(ss1,ss2)

class XHMMClustering:
    def __init__(self, initial_n_clusters = 2, n_clusters = 2, max_iter=3, llklike=False, n_quantizations=10,**kwargs):
        self.n_clusters = n_clusters
        self.initial_n_clusters = initial_n_clusters
        self.max_iter = max_iter
        self.kwargs = kwargs  
        self.labels_ = None
        self.alg = None
        self.features_train_ = None
        self._done = False
        self._llklike = llklike
        self._llkalg = None
        self._n_quantizations = n_quantizations
    
    def fit(self, data, labels = None, next_iter=False):
        if next_iter:
            labels = self.labels_
        elif labels == None:
            self.labels_ = pd.DataFrame(np.random.randint(self.initial_n_clusters, size=(data[0].shape[0],1)), index = data[0].index)
        else:
            self.labels_ =  labels   
        for i in range(self.max_iter):
            if self._done:
                continue
            alg = XHMMFeatures( n_quantizations=self._n_quantizations,**self.kwargs)
            alg.fit(data, self.labels_)
            self.alg = alg   
            features_train = alg.transform(data).dropna(axis=1)
            if self._llklike:
                self._llkalg = XHMMFeatures(self_models=True, delay_min=1, delay_max=1, n_quantizations=self._n_quantizations)
                self._llkalg.fit(data, self.labels_)
                features_train_llk = self._llkalg.transform(data).dropna(axis=1)
                features_train = pd.concat([features_train, features_train_llk], axis=1, join="outer")
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(features_train)     
            labels_ = pd.DataFrame(kmeans.labels_, index = features_train.index)
            if is_equal_labels(labels_, self.labels_):
                self._done = True
                break
            self.labels_ = labels_
        return self

    def transform(self, data):

        return self.alg.transform(data)
