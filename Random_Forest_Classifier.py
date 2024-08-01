import numpy as np
import pandas as pd
from Decision_Tree_Classifier import DecisionTreeClassifier

class RandomForestClassifier():
    def __init__(self, min_samples_split=2, max_depth=5, n_classifiers=10, info_method="entropy"):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_classifiers = n_classifiers
        self.info_method = info_method

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.dtrees = [DecisionTreeClassifier(max_depth = self.max_depth, info_method = self.info_method) for i in range(self.n_classifiers)]

        for dtree in self.dtrees:
            X_b, y_b = self._random_sample(X, y)
            dtree.fit(X_b, y_b) # fit dtree with bootstrapped training dataset
            
    def _random_sample(self, X, y):
        ''' generate and return a new bootstrapped dataset from the training data '''
        n_samples = X.shape[0]
        idx = np.random.choice(n_samples, n_samples, replace=True)
        return X[idx], y[idx]
        
    def predict(self, X):
        X = np.array(X)
        y_preds = []
        
        for x in X:
            # calculate prediction of x for every dtree
            for dtree in self.dtrees:
                y_pred = []
                y_pred.append(dtree.make_prediction(x, dtree.root))

            # take the majority vote of the n_classifiers trees
            modals, counts = np.unique(y_pred, return_counts=True)
            index = np.argmax(counts)
            y_preds.append(modals[index])

        return y_preds