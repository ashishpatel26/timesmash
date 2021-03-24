from timesmash.smash import InferredHMMLikelihood
from timesmash.distance import LikelihoodDistance
import pandas as pd
from timesmash.quantizer import Quantizer
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from timesmash.utils import process_train_labels


class ClusteredHMMClassifier:
    def __init__(
        self,
        *,
        clustering=None,
        classifier=None,
        quantizer=None,
        epsilon=0.25,
        **kwargs
    ):
        self._eps = epsilon
        self._clu = (
            KMeans(n_clusters=2, random_state=0) if clustering is None else clustering
        )
        self._qtz = Quantizer(clean=True, **kwargs) if quantizer is None else quantizer
        self._regressor = (
            RandomForestClassifier(random_state=1) if classifier is None else classifier
        )

    def produce(self, *, train, test, label):
        train, label = process_train_labels(train, label)
        qtz = Quantizer().fit(train, label=label)
        new_labels = label.copy()
        for lb, dataframe in label.groupby(label.columns[0]):
            dist = (
                LikelihoodDistance(quantizer=qtz)
                .fit(train.loc[dataframe.index])
                .produce()
            )
            sub_labels = self._clu.fit(dist).labels_
            new_label_names = [str(lb) + "_" + str(i) for i in sub_labels]
            new_labels.loc[dataframe.index, label.columns[0]] = new_label_names

        featurizer = InferredHMMLikelihood(quantizer=qtz, epsilon=self._eps)
        train_features, test_features = featurizer.fit_transform(
            train=train, test=test, label=new_labels
        )
        self._regressor.fit(train_features, label.values.ravel())
        predicted_labels = self._regressor.predict(test_features)
        return predicted_labels

    def fit(self, *args, **kwargs):
        return self
