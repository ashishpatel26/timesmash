from timesmash.featurizer import _Featurizer
from timesmash.quantizer import Quantizer
from timesmash.utils import (
    _gen_model,
    _llk,
    _llk_state,
    Binary_crashed,
    process_train_labels,
)
import pandas as pd
import numpy as np
import os
from collections import defaultdict
from abc import ABC, abstractmethod


class InferredHMMLikelihood(ABC):
    def __init__(self, *, quantizer=None, epsilon=0.25, clean=True, **kwargs):
        self.train_quatized = None
        self._fitted = False
        self.train_feature = []
        self._llk_function = _llk
        self._eps = epsilon
        self._clean = clean
        self.all_models_file = defaultdict(lambda: dict())
        self._qtz = Quantizer(clean=self._clean, **kwargs) if quantizer is None else quantizer

    def _fit(self, X, y):
        self.train_quatized = self._qtz.fit_transform(X, label=y)
        self.__fit_featurizer(X, y)
        train_quatized = self._qtz.transform(X)
        for i, data in enumerate(train_quatized):
            feature = self.get_feature(data, i)
            self.train_feature.append(feature)
        self._fitted = True
        return self

    def _transform(self, X):
        qtz_test = self._qtz.transform(X)
        feature_train = []
        train_match = self.train_feature.copy()
        for i, data in enumerate(qtz_test):
            feature = self.get_feature(data, i)
            feature_train.append(feature)
        return pd.concat(train_match, axis=1), pd.concat(feature_train, axis=1)

    def fit_transform(self, *, train, test, label):
        assert train is not None or self._fitted, "Train cannot be None."
        train, label = process_train_labels(train, label)
        test = pd.DataFrame(test)
        if train is not None:
            self._fit(train, label)
        train_features, test_features = self._transform(test)
        train_features = train_features.replace([np.inf, -np.inf], np.nan).dropna(
            axis=1
        )
        test_features = test_features.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
        common_cols = [
            col
            for col in set(train_features.columns).intersection(test_features.columns)
        ]
        assert (
            len(common_cols) != 0
        ), "No features were found, try different hyperparameters."
        train_features = train_features[common_cols].copy()
        test_features = test_features[common_cols].copy()
        train_features.index = train.index
        test_features.index = test.index
        return train_features, test_features

    def get_feature(self, dataframe, q):
        feature_list = []
        for label, models in self.all_models_file[q].items():
            try:
                feature = self._llk_function(dataframe, models, clean=self._clean)
                feature.columns = [models + str(x) for x in range(len(feature.columns))]
            except Binary_crashed as bin_error:
                continue
            except Exception as e:
                raise e
            else:
                feature_list.append(feature)
        final_df = pd.concat(feature_list, axis=1)
        return final_df

    def __fit_featurizer(self, X, y):
        for label, dataframe in y.groupby(y.columns[0]):
            quantized = self._qtz.transform(X.loc[dataframe.index])
            models = []
            for i, data in enumerate(quantized):
                try:
                    model_name = _gen_model(data, eps=self._eps, clean=self._clean)
                except Binary_crashed as e:
                    continue
                except Exception as e:
                    raise e
                else:
                    self.all_models_file[i][label] = model_name

    def __getstate__(self):
        picked_data = {}
        picked_data['class'] = self.__dict__
        picked_data['files'] = {}

        for q in range(self._qtz.get_n_quantizations()):
            for label, models in self.all_models_file[q].items():
                with open(models,"r") as f:
                    picked_data['files'][models] = f.read()
        return picked_data 
    
    
    def __setstate__(self, d):

        self.__dict__ = d['class']
        RANDOM_NAME(clean=True)
        for name, data in d['files'].items():
            with open(name,"w") as f:
                f.write(data)


class InferredHMMLikelihoodState(InferredHMMLikelihood):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._llk_function = _llk_state
