import unittest
import pandas as pd
import sys
from sklearn.ensemble import RandomForestClassifier
sys.path.append('../')
sys.path.append('./timesmash')
from timesmash import ClusteredHMMClassifier, LikelihoodDistance, SymbolicDerivative, Quantizer, InferredHMMLikelihood
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score




class Timesmashtest(unittest.TestCase):

    def test_CInferredHMMLikelihood1(
        self,
        train_path="./timesmash/example/Trace/Trace_TRAIN.tsv",
        test_path="./timesmash/example/Trace/Trace_TEST.tsv",
    ):
        train = pd.read_csv(
            train_path, header=None, delim_whitespace=True, na_values="NaN"
        )
        test = pd.read_csv(
            test_path, header=None, delim_whitespace=True, na_values="NaN"
        )
        train_label = pd.DataFrame(train[0].copy())
        del train[0]
        test_label = pd.DataFrame(test[0].copy())
        del test[0]
        clast = ClusteredHMMClassifier(n_quantizations=10)

        predicted_labels = clast.produce(
            train=train, test=test, label=train_label
        )
        accur = accuracy_score(test_label, predicted_labels)
        print('InferredHMMLikelihood: ',accur)
        self.assertTrue(accur >= 0.85)


    def test_CInferredHMMLikelihood(
        self,
        n_quantizations=10,
        n_clusters=3,
        train_path="./timesmash/example/Trace/Trace_TRAIN.tsv",
        test_path="./timesmash/example/Trace/Trace_TEST.tsv",
    ):
        train = pd.read_csv(
            train_path, header=None, delim_whitespace=True, na_values="NaN"
        )
        test = pd.read_csv(
            test_path, header=None, delim_whitespace=True, na_values="NaN"
        )

        train_label = pd.DataFrame(train[0].copy())
        del train[0]
        test_label = pd.DataFrame(test[0].copy())
        del test[0]

        _qtz = Quantizer(n_quantizations=n_quantizations)
        _qtz.fit(train, label=train_label)
        y_train = train_label.copy()
        X_train = train.copy()
        y_train_new = train_label.copy()
        for label, dataframe in y_train.groupby(y_train.columns[0]):
            clast = LikelihoodDistance(quantizer=_qtz)
            clast.fit(X_train.loc[dataframe.index])
            dist = clast.produce()
            self.assertTrue(X_train.loc[dataframe.index].shape[0] == dist.shape[0])
            self.assertTrue(dist.shape[1] == dist.shape[0])
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(dist)
            clasters = kmeans.labels_
            new_l = [str(label) + "_" + str(i) for i in clasters]
            y_train_new.loc[dataframe.index, y_train_new.columns[0]] = new_l
        clast = InferredHMMLikelihood(quantizer=_qtz, epsilon=0.01)
        train_feature, test_feature = clast.fit_transform(
            train=X_train, test=test, label=y_train_new
        )
        self.assertTrue(train_feature.shape[0] == y_train.shape[0])
        self.assertTrue(test_feature.shape[0] == train_label.shape[0])
        self.assertTrue(
            train_feature.shape[1]
            == n_quantizations * n_clusters * train_label[0].unique().shape[0]
        )

        clf = RandomForestClassifier(random_state=1)
        clf.fit(train_feature, train_label.values.ravel())
        predicted_labels = clf.predict(test_feature)
        accur = accuracy_score(test_label, predicted_labels)
        print('CInferredHMMLikelihood: ',accur)
        self.assertTrue(accur >= 0.85)

    def test_SymbolicDerivative(
        self,
        train_path="./timesmash/example/Trace/Trace_TRAIN.tsv",
        test_path="./timesmash/example/Trace/Trace_TEST.tsv",
    ):
        train = pd.read_csv(
            train_path, header=None, delim_whitespace=True, na_values="NaN"
        )
        test = pd.read_csv(
            test_path, header=None, delim_whitespace=True, na_values="NaN"
        )
        train_label = pd.DataFrame(train[0].copy())
        del train[0]
        test_label = pd.DataFrame(test[0].copy())
        del test[0]
        clast = SymbolicDerivative(n_quantizations=10)
        train_feature, test_feature = clast.fit_transform(
            train=train, test=test, label=train_label
        )
        clf = RandomForestClassifier(random_state=1)
        clf.fit(train_feature, train_label.values.ravel())
        predicted_labels = clf.predict(test_feature)
        accur = accuracy_score(test_label, predicted_labels)
        print('SymbolicDerivative: ',accur)
        self.assertTrue(accur >= 0.98)


if __name__ == "__main__":
    unittest.main()
