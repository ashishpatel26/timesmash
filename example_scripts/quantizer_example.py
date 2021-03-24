import pandas as pd
import sys
import numpy as np
from timesmash import Quantizer
from timesmash import XG2 as fetur
from timesmash import Lsmash_distance
from timesmash.cynet import xgModels
from timesmash.utils import genesess

data1 = pd.DataFrame(np.random.randn(100, 100))
data2 = pd.DataFrame(np.random.randn(5, 100))
l = list()
for i in range(100):
    l.append(i % 4)
label = pd.DataFrame(l)
_qtz = Quantizer()
train_quatized = _qtz.fit_transform(data1, labels=label)
test_quatized = _qtz.transform(data2)
print(train_quatized)
print(test_quatized)
print(_qtz.get_n_quantizations())
"""
data.ix[0].index.to_series().to_csv("index", index=False, header=False)
xg = xgModels( TS_PATH="./data.dat",
                NAME_PATH="./index",
                LOG_PATH="./log.dat",
                FILEPATH="./m",
                BEG=0,
                END=1,
                NUM=20,
                PARTITION=[0],
                XgenESeSS_PATH="./timesmash/bin/XgenESeSS_FOR",
                RUN_LOCAL = True)
xg.run(target = [0])
test = pd.DataFrame(np.random.randn(51, 100))
l = list()
for i in range(26):
	l.append (i%4)
label= pd.DataFrame(l)
#label = None
test.index = list(range(150,201))
n_quant = 1
di = Lsmash_distance(5)
di.fit(data,labels=label)
print(di.produce(test))"""
