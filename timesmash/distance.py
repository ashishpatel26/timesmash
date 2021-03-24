import pandas as pd
import sys
import numpy as np
from abc import ABC, abstractmethod
from timesmash.quantizer import Quantizer
from timesmash.utils import smash, _lsmash, process_train_labels
import sklearn.cluster


class _Distance(ABC):
    def __init__(self, n_quantizations=1, *, quantizer=None, clean=True, **kwargs):
        self.data = None
        self.n_quantizations = n_quantizations
        self.train_quatized = None
        self._clean = clean
        self._qtz = Quantizer(clean=self._clean, **kwargs) if quantizer is None else quantizer

    def fit(self, train, *, label=None):
        train, label = process_train_labels(train, label)
        self.train_quatized = self._qtz.fit_transform(
            train, label=label, force_refit=False
        )
        return self

    def produce(self, test=None, *, average=True):
        if test is not None:
            test = pd.DataFrame(test)
            test_quatized = self._qtz.transform(test)
        dist = []

        for i in range(self._qtz.get_n_quantizations()):
            if test is not None:

                distance_matrix = self.get_distance(
                    pd.concat([next(self.train_quatized), next(test_quatized)])
                )

            else:
                distance_matrix = self.get_distance(next(self.train_quatized))

            if distance_matrix is not None:

                if average:
                    dist.append(distance_matrix)
                else:
                    print("yield")
                    # yield distance_matrix

        if average:
            df_average= pd.concat(dist).groupby(level=0).mean()
            df_average = df_average[dist[0].columns]
            df_average = df_average.loc[dist[0].columns]
            return df_average

    @abstractmethod
    def get_distance(self, dataframe):
        pass


class Gsmash_distance(_Distance):
    def get_distance(self, dataframe):
        return smash(dataframe, clean=self._clean)


class LikelihoodDistance(_Distance):
    def get_distance(self, dataframe):
        return _lsmash(dataframe, clean=self._clean)
