import unittest
import pandas as pd
import sys
from sklearn.ensemble import RandomForestClassifier

sys.path.append("../")
sys.path.append("./")
from timesmash import *
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

train_path = "./notebooks/Trace/Trace_TRAIN.tsv"
test_path = "./notebooks/Trace/Trace_TEST.tsv"


class Timesmashtest(unittest.TestCase):
    
    def test_XHMMFeatures(self):

        d1_train = pd.DataFrame(
            [[0, 1, 0, 1, 0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]],
            index=["person_1", "person_2"],
        )
        d2_train = pd.DataFrame(
            [[1, 0, 1, 0, 1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]],
            index=["person_1", "person_2"],
        )
        labels = pd.DataFrame([0, 1], index=["person_1", "person_2"])

        alg = XHMMFeatures(n_quantizations=1)
        features_train = alg.fit_transform([d1_train, d2_train], labels)

        clf = RandomForestClassifier()
        clf.fit(features_train, labels)

        d1_test = pd.DataFrame([[1, 0, 1, 0, 1, 0, 1, 0, 1]], index=["person_test"])
        d2_test = pd.DataFrame([[0, 1, 0, 1, 0, 1, 0, 1, 0]], index=["person_test"])

        features_test = alg.transform([d1_test, d2_test])
        test_label = clf.predict(features_test)
        self.assertTrue(test_label[0] == 0)

    def test_CInferredHMMLikelihood1(self):
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

        predicted_labels = clast.produce(train=train, test=test, label=train_label)
        accur = accuracy_score(test_label, predicted_labels)
        self.assertTrue(accur >= 0.75)

    def test_CInferredHMMLikelihood(self, n_quantizations=10, n_clusters=3):
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
        self.assertTrue(accur >= 0.85)

    def test_SymbolicDerivative(self):
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
        self.assertTrue(accur >= 0.98)

    def test_LikelihoodDistance(self):

        train = [
            [1, 0, 1, 0, 1, 0],
            [1, 1, 0, 1, 1, 0],
            [0, 1, 0, 1, 0, 1],
            [0, 1, 1, 0, 1, 1],
        ]
        dist_calc = LikelihoodDistance().fit(train)
        dist = dist_calc.produce()
        from sklearn.cluster import KMeans

        clusters = KMeans(n_clusters=2).fit(dist).labels_
        self.assertTrue(clusters[0] == clusters[2])
        self.assertFalse(clusters[0] == clusters[1])

    def test_XHMMClustering(self):

        channel1 = pd.DataFrame(
            [
                [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            ],
            index=["person_1", "person_2", "person_3", "person_4"],
        )
        channel2 = pd.DataFrame(
            [
                [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            ],
            index=["person_1", "person_2", "person_3", "person_4"],
        )
        alg = XHMMClustering(n_quantizations=1,llklike=True).fit(
            [channel1, channel2]
        )
        clusters = alg.labels_
        self.assertTrue(clusters.loc["person_1",0] == clusters.loc["person_3",0])
        self.assertFalse(clusters.loc["person_1",0] == clusters.loc["person_2",0])

    def test_SymbolicDerivative_produce(self):
        train = [[1, 0, 1, 0, 1, 0], [1, 1, 0, 1, 1, 0]]
        test = [[0, 1, 1, 0, 1, 1]]
        train_label = [0, 1]
        alg = SymbolicDerivative().fit(
            train=train, label=train_label
        )
        train_features = alg.transform(train)
        test_features = alg.transform(test)
        clf = RandomForestClassifier().fit(train_features, train_label)
        label = clf.predict(test_features)
        self.assertTrue(label[0] == 1)

if __name__ == "__main__":
    unittest.main()
