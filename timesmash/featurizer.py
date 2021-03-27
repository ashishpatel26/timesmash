import os
from timesmash.quantizer import Quantizer
from timesmash.utils import xgenesess, genesess, Binary_crashed, process_train_labels
import pandas as pd
from abc import abstractmethod
import warnings
import numpy as np

class Get_Fit_Transform():

    def fit_transform(self, *, train, test, label=None):
        self.fit(train, label)

        train_features = self.transform(train, warn=False)
        test_features = self.transform(test, warn=False)
        train_features = train_features.replace([np.inf, -np.inf], np.nan).dropna(
            axis=1
        )
        test_features = test_features.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
        common_cols = [
            col
            for col in set(train_features.columns).intersection(test_features.columns)
        ]
        return train_features[common_cols], test_features[common_cols]

class _Featurizer(Get_Fit_Transform):

    """
    _Featurizer abstract class is used to implement XG1 and XG2. 
    @author zed.uchicago.edu
    Inputs:
        quantizer (class Quantizer): fitted qunatizer
        **kwargs : key word arguments that are passed to Quantizer 
                    Ignored if quantizer argument is not None
    """

    def __init__(self, *, clean=True, quantizer=None, **kwargs):

        self._fitted = quantizer is not None
        self.train_feature = []
        self.clean = clean
        self._qtz = Quantizer(clean=clean,return_failed=False, **kwargs) if quantizer is None else quantizer
        assert quantizer is None or quantizer.IsFitted(), "Initialize with fitted quantizer"

    def fit(self, train, label=None):

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
        train, label = process_train_labels(train, label)
        if not self._fitted:
            self.train_quatized = self._qtz.fit(train, label=label)
            self._fitted = True
        else:
            warnings.warn("Nothing done in fit. Already fitted or initialized with quantizer")

        return self

    def transform(self, X, warn=True):
        X = pd.DataFrame(X)
        train_quatized = self._qtz.transform(X)
        crash = False
        features = []
        for index, df in enumerate(train_quatized):
            # check that binary didn't fail
            try:
                feature = self._get_feature(df, clean=self.clean)
                feature.columns = [
                    str(index) + "_" + str(x) for x in range(len(feature.columns))
                ]
            except Binary_crashed as e:
                crash = True
                feature = None
                continue
            except Exception as e:
                raise e
            features.append(feature)

        final = pd.concat(features, axis=1)

        if (final.isnull().values.any() or crash) and warn:
            warnings.warn("Quantization issues detected. Try using fit_transform to avoid NaN")
        return final

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

