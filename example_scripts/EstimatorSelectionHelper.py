import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


np.random.seed()
import random
random.seed()

def mod_acur_func(actual, pred):
    mod_actual = [int(i.split('_')[0]) for i in actual]
    mod_pred = [int(i.split('_')[0]) for i in pred]
    return accuracy_score(mod_actual, mod_pred)

mod_scorer = make_scorer(mod_acur_func)


class EstimatorSelectionHelper:

    def __init__(self, cv=10, n_jobs=-1, verbose=0, scoring='accuracy', refit=True):
        self.models = {
        'RandomForestClassifier': RandomForestClassifier(random_state=1),
        'AdaBoostClassifier': AdaBoostClassifier(random_state=1),
        'GradientBoostingClassifier': GradientBoostingClassifier(random_state=1),
        'SVC': SVC(random_state=1)}
        self.params = {
        'RandomForestClassifier': {"max_depth": [3, None],
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]},
        'AdaBoostClassifier':  {'base_estimator': [#tree.DecisionTreeClassifier(max_depth=3),
                               tree.ExtraTreeClassifier(max_depth=4)],
#            'learning_rate': [0.01, 0.1, 0.5, 1.],
#            'n_estimators': [5, 10, 15, 20, 50, 75],
            'algorithm': ['SAMME', 'SAMME.R']},
        'GradientBoostingClassifier': {'learning_rate': [0.8, 1.0] },
        'SVC': [
            {'kernel': ['linear'], 'C': [1, 10]},
            {'kernel': ['rbf'], 'C': [1, 10], 'gamma': [0.001, 0.1, 1]},
#            {'kernel': ['poly'], 'degree': [2, 3], 'gamma': [0.001, 0.1, 1]},
        ]
    }
        self.keys = self.models.keys()
        self.grid_searches = {}
        self.best = 0
        self.best_grid = None
        self.best_model = None 
        self.best_params = None
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.scoring = scoring
        self.refit = refit

    def fit(self, X, y):
        max_cv = min(y[y.columns[0]].value_counts().min(), self.cv)
        if max_cv == 1:
            self.best_grid = RandomForestClassifier(random_state=1)
            self.best_grid.fit(X,y)
            self.best = 0
            print("Cannot cross validate. Min number of timeseriaces in a label equal 1")            
        else:
            for key in self.keys:
                #print("Running GridSearchCV for %s." % key)
                model = self.models[key]
                params = self.params[key]
                if max_cv !=  self.cv:
                    print("Cross validation = " + str(self.cv) + ", too high for the dataset.")
                    print("Running with cross validation = " + str(max_cv))
                gs = GridSearchCV(model, params, cv=max_cv, n_jobs=self.n_jobs,
                                  verbose=self.verbose, scoring=self.scoring, refit=self.refit,
                                  return_train_score=True, error_score = 0)
                gs.fit(X,y)
                self.grid_searches[key] = gs
                if (gs.best_score_ > self.best):
                    self.best_grid = gs
                    self.best = gs.best_score_                 

    def score_summary(self, sort_by='mean_score'):
        def row(key, scores, params):
            d = {
                 'estimator': key,
                 'min_score': min(scores),
                 'max_score': max(scores),
                 'mean_score': np.mean(scores),
                 'std_score': np.std(scores),
            }
            return pd.Series({**params,**d})

        rows = []
        for k in self.grid_searches:
            print(k)
            params = self.grid_searches[k].cv_results_['params']
            scores = []
            for i in range(self.grid_searches[k].cv):
                key = "split{}_test_score".format(i)
                r = self.grid_searches[k].cv_results_[key]        
                scores.append(r.reshape(len(params),1))

            all_scores = np.hstack(scores)
            for p, s in zip(params,all_scores):
                rows.append((row(k, s, p)))

        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)

        columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]

        return df[columns]
    
    def predict(self, b):
       
        return self.best_grid.predict(b)

def get_path(dataset,directory = '/home/virotaru/data_smashing_/timesmash/UCRArchive_2018'):
    dataset_path = directory + '/' + dataset +'/' + dataset
    return dataset_path + '_TRAIN.tsv',  dataset_path + '_TEST.tsv'
