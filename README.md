<img src="http://zed.uchicago.edu/logo/logozed1.png" alt="logo" width="500" />

# Data Smashing

Quantifier of universal similarity amongst arbitrary data streams without a priori knowledge, features, or training.

Can be used to solve time series clustering and classification problems.

Featurization algorithms: SymbolicDerivative, InferredHMMLikelihood, Csmash
	
Distance measure: LikelihoodDistance

For questions or suggestions contact:zed@uchicago.edu

##	Usage examples	
### SymbolicDerivative
	from timesmash import SymbolicDerivative
	from sklearn.ensemble import RandomForestClassifier

	train = [[1, 0, 1, 0, 1, 0], [1, 1, 0, 1, 1, 0]]
	train_label = [[0], [1]]
	test = [[0, 1, 1, 0, 1, 1]]
	train_features, test_features = SymbolicDerivative().fit_transform(
	    train=train, test=test, label=train_label
	)
	clf = RandomForestClassifier().fit(train_features, train_label)
	label = clf.predict(test_features)
	print("Predicted label: ", label)
	
###	LikelihoodDistance	
	from timesmash import LikelihoodDistance
	from sklearn.cluster import KMeans
	train = [[1, 0, 1.1, 0, 11.2, 0], [1, 1, 0, 1, 1, 0], [0, 0.9, 0, 1, 0, 1], [0, 1, 1, 0, 1, 1]]
	dist_calc = LikelihoodDistance().fit(train)
	dist = dist_calc.produce()
	from sklearn.cluster import KMeans
	clusters = KMeans(n_clusters = 2).fit(dist).labels_
	print("Clusters: " clusters)
	
###	InferredHMMLikelihood	
	from timesmash import InferredHMMLikelihood
	from sklearn.ensemble import RandomForestClassifier

	train = [[1, 0, 1, 0, 1, 0], [1, 1, 0, 1, 1, 0]]
	train_label = [[0], [1]]
	test = [[0, 1, 1, 0, 1, 1]]
	train_features, test_features = InferredHMMLikelihood().fit_transform(
	    train=train, test=test, label=train_label
	)
	clf = RandomForestClassifier().fit(train_features, train_label)
	label = clf.predict(test_features)
	print("Predicted label: ", label)

###	ClusteredHMMClassifier:	
	from timesmash import Quantizer, InferredHMMLikelihood, LikelihoodDistance
	from sklearn.cluster import KMeans
	from sklearn.ensemble import RandomForestClassifier
	import pandas as pd

	train = pd.DataFrame(
	    [[1, 0, 1, 0, 1, 0], [1, 1, 0, 1, 1, 0], [1, 0, 1, 0, 1, 0], [1, 1, 0, 1, 1, 0]]
	)
	train_label = pd.DataFrame([[0], [1], [0], [1]])
	test = pd.DataFrame([[0, 1, 1, 0, 1, 1]])

	qtz = Quantizer().fit(train, label=train_label)
	new_labels = train_label.copy()
	for label, dataframe in train_label.groupby(train_label.columns[0]):
	    dist = LikelihoodDistance(quantizer=qtz).fit(train.loc[dataframe.index]).produce()
	    sub_labels = KMeans(n_clusters=2, random_state=0).fit(dist).labels_
	    new_label_names = [str(label) + "_" + str(i) for i in sub_labels]
	    new_labels.loc[dataframe.index, train_label.columns[0]] = new_label_names

	featurizer = InferredHMMLikelihood(quantizer=qtz, epsilon=0.01)
	train_features, test_features = featurizer.fit_transform(
	    train=train, test=test, label=new_labels
	)

	clf = RandomForestClassifier().fit(train_features, train_label)
	print("Predicted label: ", clf.predict(test_features))

###	XHMMFeatures for anomaly detection:	
	import pandas as pd
	from timesmash import XHMMFeatures
	from sklearn.neighbors import LocalOutlierFactor

	channel1_train = pd.DataFrame([[0,1,0,1,0,1,0,1,0,1],[1,0,1,0,1,0,1,0,1,0]], index=['person_1', 'person_2'])
	channel2_train = pd.DataFrame([[0,1,0,1,0,1,0,1,0,1],[1,0,1,0,1,0,1,0,1,0]], index=['person_1', 'person_2'])
	labels = pd.DataFrame([1,1], index=['person_1', 'person_2'])
	    
	alg = XHMMFeatures(n_quantizations=1)
	features_train = alg.fit_transform([channel1_train,channel2_train], labels)
	    
	clf = LocalOutlierFactor(novelty=True)  
	clf.fit(features_train)
	        
	channel1_test = pd.DataFrame([[0,1,0,1,0,1,0,1,0,1],[1,0,1,0,1,0,1,0,1]], index=['person_test_1', 'person_test_2'])
	channel2_test= pd.DataFrame([[0,1,0,1,0,1,0,1,0,1],[0,1,0,1,0,1,0,1,0]], index=['person_test_1', 'person_test_2'])

	features_test = alg.transform([channel1_test,channel2_test])
	print(clf.predict(features_test))

###	XHMMFeatures for classification:	
	import pandas as pd
	from timesmash import XHMMFeatures
	from sklearn.ensemble import RandomForestClassifier

	d1_train = pd.DataFrame([[0,1,0,1,0,1,0,1,0,1],[1,0,1,0,1,0,1,0,1,0]], index=['person_1', 'person_2'])
	d2_train = pd.DataFrame([[1,0,1,0,1,0,1,0,1,0],[1,0,1,0,1,0,1,0,1,0]], index=['person_1', 'person_2'])
	labels = pd.DataFrame([0,1], index=['person_1', 'person_2'])
	    
	alg = XHMMFeatures(n_quantizations=1)
	features_train = alg.fit_transform([d1_train,d2_train], labels)
	    
	clf = RandomForestClassifier()  
	clf.fit(features_train, labels)
	        
	d1_test = pd.DataFrame([[1,0,1,0,1,0,1,0,1]], index=['person_test'])
	d2_test= pd.DataFrame([[0,1,0,1,0,1,0,1,0]], index=['person_test'])

	features_test = alg.transform([d1_test,d2_test])
	    
	print(clf.predict(features_test))