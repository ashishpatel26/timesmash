import os
from timesmash.quantizer import Quantizer
from timesmash.utils import xgenesess, genesess, Binary_crashed, process_train_labels
import pandas as pd
from abc import ABC, abstractmethod


class _Featurizer(ABC):

    """
    _Featurizer abstract class is used to implement XG1 and XG2. 
    @author zed.uchicago.edu
    Inputs:
        quantizer (class Quantizer): fitted qunatizer
        **kwargs : key word arguments that are passed to Quantizer 
                    Ignored if quantizer argument is not None
    """

    def __init__(self, *, clean=True, quantizer=None, **kwargs):
        self._fitted = False
        self.train_feature = []
        self.clean = clean
        self._qtz = Quantizer(clean=clean,return_failed=False, **kwargs) if quantizer is None else quantizer

    def fit(self, X, y=None):

        """
        @author zed.uchicago.edu
        Fit method that fits the Qunatizer if not fitted and feturizes 
                        train data. Use fit_transform if test data is available. 
        Inputs:
            X (pandas.DataFrame): train data
            y (pandas.DataFrame): optional labels for train data
        Outputs:
            pd.Dataframe of time series features 
        """

        self.train_quatized = self._qtz.fit_transform(X, label=y)
        for index, df in enumerate(self.train_quatized):
            # check that binary didn't fail
            try:
                feature = self._get_feature(df, clean=self.clean)
                feature.columns = [
                    str(index) + "_" + str(x) for x in range(len(feature.columns))
                ]
            except Binary_crashed as e:
                feature = None
                continue
            except Exception as e:
                raise e
            self.train_feature.append(feature)
        self._fitted = True
        return pd.concat(self.train_feature, axis=1)

    def _predict(self, X):

        """
        @author zed.uchicago.edu
        Private method that quantizes and featurizes test data and removes qunatization from 
                        test data if it failed in test. 
        Inputs:
            X (pandas.DataFrame): test data
        Outputs:
            two pd.Dataframe of time series features for test and train
        """

        assert self._fitted, "Object not fitted!"
        qtz_test = self._qtz.transform(X)
        feature_test = []
        train_match = self.train_feature.copy()
        train_index = None
        test_index = None
        for i, data in enumerate(qtz_test):
            if train_match[i] is None:
                feature_test.append(None)
                continue
            try:
                feature = self._get_feature(data, clean=self.clean)
                feature.columns = [
                    str(i) + "_" + str(x) for x in range(len(feature.columns))
                ]
            except Binary_crashed as e:
                train_match[i] = None
                continue
            except Exception as e:
                raise e
            feature_test.append(feature)
            train_index = train_match[i].index
            test_index = feature.index
        train_features = pd.concat(train_match, axis=1, ignore_index=False)
        test_features = pd.concat(feature_test, axis=1, ignore_index=False)
        common_cols = [
            col
            for col in set(train_features.columns).intersection(test_features.columns)
        ]
        return train_features[common_cols], test_features[common_cols]

    def fit_transform(self, *, train=None, test, label=None):
        test = pd.DataFrame(test)
        train, label = process_train_labels(train, label)
        if not self._fitted:
            self.fit(train, label)
        return self._predict(test)

    @abstractmethod
    def _get_feature(self, dataframe):
        pass


class XG1(_Featurizer):
    def __init__(self, *, max_delay=20, min_delay=0, **kwargs):
        super().__init__(**kwargs)
        self.max_delay = max_delay
        self.min_delay = min_delay

    def _get_feature(self, dataframe, **kwargs):
        return xgenesess(
            dataframe, max_delay=self.max_delay, min_delay=self.min_delay, **kwargs
        )


class SymbolicDerivative(_Featurizer):
    def __init__(self, *, depth=100, epsilon=0.25, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._eps = epsilon

    def _get_feature(self, dataframe, **kwargs):
        return genesess(
            dataframe,
            multi_line=True,
            depth=self._depth,
            gen_epsilon=self._eps,
            **kwargs
        )
