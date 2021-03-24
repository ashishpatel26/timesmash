import pandas as pd
import sys
import time
sys.path.append("../")
"""

from timesmash import SymbolicDerivative, ClusteredHMMClassifier, InferredHMMLikelihoodState
from sklearn.ensemble import RandomForestClassifier
train = [[5, 0, 5, 0, 5, 0], [5, 5, 0, 5, 5, 0]]

train_label = [[0], [5]]
test = [[5, 5, 5, 5, 5, 3]]
print(pd.DataFrame(test))
train_features, test_features = SymbolicDerivative(n_quantizations = 1, clean =True).fit_transform(
    train=train, test=test, label=None
)
print(train_features.shape)
print(test_features.shape)
clf = RandomForestClassifier(random_state=5).fit(train_features, train_label)
label = clf.predict(test_features)
print("Predicted label", label)


import pandas as pd
import sys

sys.path.append("../")
from timesmash import InferredHMMLikelihood
from sklearn.ensemble import RandomForestClassifier

train = [[5, 0, 5, 0, 5, 0], [5, 5, 0, 5, 5, 0, 5, 5, 0]]

train_label = [[0], [5]]
test = [[0, 5, 5, 0, 5, 5, 0], [0, 5, 5, 0, 5, 5, 0, 0, 5, 5, 0, 5, 5, 0]]
train_features, test_features = InferredHMMLikelihoodState(clean = False, n_quantizations = 2).fit_transform(
    train=train, test=test, label=train_label
)
print(train_features.shape)
print(test_features.shape)
clf = RandomForestClassifier(random_state=5).fit(train_features, train_label)
label = clf.predict(test_features)
print("Predicted label", label)

from timesmash import Quantizer, InferredHMMLikelihood, LikelihoodDistance
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier


from timesmash import ClusteredHMMClassifier

train = [[5, 0, 5, 0, 5, 0], [5, 5, 0, 5, 5, 0], [0, 5, 0, 5, 0, 5], [0, 5, 5, 0, 5, 5]]
train_label = [[0], [5], [0], [5]]
test = [[0, 5, 5, 0, 5, 5]]

print(ClusteredHMMClassifier().produce(train=train, label=train_label, test=test))


train = pd.DataFrame(
    [[5, 0, 5, 0, 5, 0], [5, 5, 0, 5, 5, 0], [0, 5, 0, 5, 0, 5], [0, 5, 5, 0, 5, 5]]
)
train_label = pd.DataFrame([[0], [5], [0], [5]])
test = pd.DataFrame([[0, 5, 5, 0, 5, 5]])


qtz = Quantizer().fit(train, label=train_label)
# qtz.set_quantization([0.3, 0.5])
new_labels = train_label.copy()
for label, dataframe in train_label.groupby(train_label.columns[0]):
    dist = LikelihoodDistance(quantizer=qtz).fit(train.loc[dataframe.index]).produce()
    sub_labels = KMeans(n_clusters=2, random_state=0).fit(dist).labels_
    new_label_names = [str(label) + "_" + str(i) for i in sub_labels]
    new_labels.loc[dataframe.index, train_label.columns[0]] = new_label_names

featurizer = InferredHMMLikelihood(quantizer=qtz, epsilon=0.05)
train_features, test_features = featurizer.fit_transform(
    train=train, test=test, label=new_labels
)

# utilizing timeInferredHMMLikelihood features in a classifier
clf = RandomForestClassifier().fit(train_features, train_label)
print("Predicted label", clf.predict(test_features))
"""
from timesmash import LikelihoodDistance
'''
train = [
    [5, 0, 5.5, 0, 5.2, 0],
    [5, 5, 0, 5, 5, 0],
    [0, 0.9, 0, 5, 0, 5],
    [0, 5, 5, 0, 5, 5],
]


dist_calc = LikelihoodDistance().fit(train)
dist = dist_calc.produce()

# utilizing Kmeans for clustering
from sklearn.cluster import KMeans

clusters = KMeans(n_clusters=2).fit(dist).labels_
print(clusters)
'''
from timesmash import LikelihoodDistance
from timesmash.utils import _lsmash
train = pd.DataFrame([
    [1, 0, 1, 0, 1, 0],
    [1, 1, 0, 1, 1, 0],
    [0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0],
])


from timesmash import LikelihoodDistance
from timesmash.utils import _lsmash
dist = _lsmash(train, clean=False)
print(dist)
# utilizing Kmeans for clustering
from sklearn.cluster import KMeans

clusters = KMeans(n_clusters=2).fit(dist).labels_
print(clusters)

#
