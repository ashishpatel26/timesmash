#!/usr/bin/env python3
import pandas as pd
import sys
import numpy as np
import pickle
from timesmash import Gsmash_featurizer, Lsmash_distance
import warnings
from sklearn.manifold import (
    LocallyLinearEmbedding,
    TSNE,
    Isomap,
    MDS,
    TSNE,
    smacof,
    SpectralEmbedding,
)
from sklearn.metrics import accuracy_score
from EstimatorSelectionHelper import EstimatorSelectionHelper, get_path

# https://archive.ics.uci.edu/ml/datasets/Synthetic+Control+Chart+Time+Series


dataset = sys.argv[1]
train_path, test_path = get_path(dataset)
print(train_path)
train = pd.read_csv(train_path, header=None, delim_whitespace=True, na_values="NaN")
test = pd.read_csv(test_path, header=None, delim_whitespace=True, na_values="NaN")
test.index = test.index + train.shape[0]
train_label = pd.DataFrame(train[0].copy())
del train[0]
test_label = pd.DataFrame(test[0].copy())
del test[0]
clast = Lsmash_distance()
clast.fit(train, labels=train_label)
dist = clast.produce(test)
best_score = 0
best_grid = None
best_emb = None
best_b = None
best_n = 0
all_emb = [LocallyLinearEmbedding, Isomap, MDS, SpectralEmbedding]

if True:
    for n in [5, 10, 20, 40]:
        for emb in all_emb:
            try:
                embedding = emb(n_components=n)
            except:
                embedding = emb()
            X_transformed = embedding.fit_transform(dist)
            X_transformed = pd.DataFrame(X_transformed)
            train_feat = X_transformed.loc[train.index]
            test_feat = X_transformed.loc[test.index]
            helper1 = EstimatorSelectionHelper()
            helper1.fit(train_feat, train_label)
            best_score_ = helper1.best
            if best_score_ >= best_score:
                best_score = best_score_
                best_grid = helper1.best_grid
                best_emb = embedding
                best_b = test_feat
                best_n = n
                helper_predict = best_grid.predict(best_b)
                print("n features: " + str(best_n))
                print(best_emb)
                print(helper1.best_grid)
                accur = accuracy_score(test_label, helper_predict)
                print("\nhelper_score so far :")
                print(accur)

helper_predict = best_grid.predict(best_b)
print("n features: " + str(best_n))
print(best_emb)
print(helper1.best_grid)
accur = accuracy_score(test_label, helper_predict)
print("\nhelper_score so far :")
print(accur)
