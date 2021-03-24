import pandas as pd
import sys
import numpy as np
from timesmash import Lsmash_distance
from EstimatorSelectionHelper import get_path

dataset = sys.argv[1]
train_path, test_path = get_path(dataset)
print(train_path)
train = pd.read_csv(train_path, header=None, delim_whitespace=True, na_values="NaN")
test = pd.read_csv(test_path, header=None, delim_whitespace=True, na_values="NaN")

train_label = pd.DataFrame(train[0].copy())
del train[0]
test_label = pd.DataFrame(test[0].copy())
del test[0]

clast = Lsmash_distance()
clast.fit(train)
clast.produce(test)
