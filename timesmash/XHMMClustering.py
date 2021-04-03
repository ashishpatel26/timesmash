import pandas as pd
import numpy as np
from timesmash import Quantizer, XHMMFeatures, InferredHMMLikelihood
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
    def __init__(self, initial_n_clusters = 2, n_clusters = 2, max_iter=3, llklike=False, **kwargs):
        self.n_clusters = n_clusters
        self.initial_n_clusters = initial_n_clusters
        self.max_iter = max_iter
        self.kwargs = kwargs  
        self.labels_ = None
        self.alg = None
        self.done = False
        self._llklike = llklike
        self._llkalg = None
        self.features = None
        self.kwargs_llk = kwargs.copy()
        self.kwargs_llk['self_models'] = True
        self.kwargs_llk['delay_min'] = 1
        self.kwargs_llk['delay_max'] = 1

    
    def fit(self, data, labels = None):
        if labels is None:
            self.labels_ = pd.DataFrame(np.random.randint(self.initial_n_clusters, size=(data[0].shape[0],1)), index = data[0].index)
        else:
            self.labels_ =  labels   
        for i in range(self.max_iter):
            self.alg = XHMMFeatures(**self.kwargs)
            self.alg.fit(data, self.labels_)
            if self._llklike:
                self._llkalg = XHMMFeatures(**self.kwargs_llk)
                self._llkalg.fit(data, self.labels_)   
            self.features = self.transform(data).dropna(axis=1)
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(self.features)     
            labels_ = pd.DataFrame(kmeans.labels_, index = self.features.index)
            if is_equal_labels(labels_, self.labels_):
                self.done = True
                break
            self.labels_ = labels_
        return self

    def transform(self, data):
        features = self.alg.transform(data).dropna(axis=1)
        if self._llklike:
            features_llk = self._llkalg.transform(data)
            features = pd.concat([features, features_llk], axis=1, join="outer")

        return features

