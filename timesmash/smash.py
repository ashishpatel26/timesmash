from timesmash.quantizer import Quantizer
from timesmash.utils import (
    _gen_model,
    _llk,
    _llk_state,
    Binary_crashed,
    process_train_labels,
)
from timesmash.featurizer import Get_Fit_Transform
import pandas as pd
from collections import defaultdict
import warnings


class InferredHMMLikelihood(Get_Fit_Transform):
    def __init__(self, *, quantizer=None, epsilon=0.25, clean=True, **kwargs):
        self._fitted = False
        self.train_feature = []
        self._llk_function = _llk
        self._eps = epsilon
        self._clean = clean
        self.all_models_file = defaultdict(lambda: dict())
        self._qtz = Quantizer(clean=self._clean, **kwargs) if quantizer is None else quantizer
        assert quantizer is None or quantizer.IsFitted(), "Initialize with fitted quantizer"


    def fit(self, train, label):
        train, label = process_train_labels(train, label)
        if not self._qtz.IsFitted():
            self._qtz.fit(train, label=label)
        self.__fit_featurizer(train, label)
        return self

    def transform(self, X, warn=True):
        X = pd.DataFrame(X)
        qtz_test = self._qtz.transform(X)
        feature_train = []
        for i, data in enumerate(qtz_test):
            feature = self._get_feature(data, i, warn = warn)
            feature_train.append(feature)
        return pd.concat(feature_train, axis=1)

    def _get_feature(self, dataframe, q, warn):
        feature_list = []
        crash = False
        for label, models in self.all_models_file[q].items():
            try:
                feature = self._llk_function(dataframe, models, clean=self._clean)
                feature.columns = [models + str(x) for x in range(len(feature.columns))]
            except Binary_crashed as bin_error:
                crash = True
                continue
            except Exception as e:
                raise e
            else:
                feature_list.append(feature)

        final_df = pd.concat(feature_list, axis=1)
        if (final_df.isnull().values.any() or crash) and warn:
            warnings.warn("Quantization issues detected. Try using fit_transform to avoid NaN")
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
