import numpy as np
import pandas as pd
from Decision_Tree_Classifier import DecisionTreeClassifier

class AdaBoostClassifier:
    def __init__(self, k_classifiers):
        self.k_classifiers = k_classifiers
        self.trees = [] # k trees
        self.say = [] # the a_k values for each of the k trees
        
    def fit(self, X, y):
        ''' train an AdaBoost forest on a dataset '''
        X = np.array(X)
        y = np.array(y)
        self.n_classes = np.sum(np.unique(y))
        
        N = X.shape[0]
        weights = np.array([(1 / N) for _x_ in range(N)]) # Initialize uniform importance to each example
        
        # sample_ranges is the cumulative sum array of weights
        # it is used by _create_weighted_dataset to translate sample weights into interval ranges to proxy for their probabilities during selection
        self.sample_ranges = list(np.cumsum(weights))
        self.sample_ranges[-1] = np.float64(1)
        
        for k in range(self.k_classifiers):
            # Train kth stump on data
            tree = DecisionTreeClassifier(max_depth=1) 
            tree.fit(pd.DataFrame(X), pd.DataFrame(y))
            y_preds = tree.predict(X)

            # calculate total error (sum of weights of wrongly classified samples)
            incorrect_classifications_weights = [weights[n] for n in range(N) if y_preds[n] != y[n]]
            total_error = np.sum(incorrect_classifications_weights) 
            
            if total_error == 0:
                total_error += .0001 # have to add a small value to avoid divide by 0
            if total_error == 1:
                total_error -= .0001 # have to add a small value to avoid divide by 0
            
            # calculate "amount of say" / "adaptive parameter" / a_k for this tree
            a_k = 0.5 * np.log((1 - total_error) / total_error)
            self.say.append(a_k) # add it to the list, we will need each of them when making predictions

            # weight the data according to a_k 
            for n in range(N):
                # increase weights for incorrectly classified samples
                if y_preds[n] != y[n]:
                    weights[n] = weights[n] * np.exp(a_k)
                # decrease weights for correctly classified samples
                elif y_preds[n] == y[n]: 
                    weights[n] = weights[n] * np.exp(-a_k)
            
            # normalize weights so that total probability remains 1
            normalizer = np.sum(weights)
            weights = weights / normalizer  
            
            # weight the dataset for the next tree
            X, y = self._create_weighted_dataset(X, y, weights)
            
            self.trees.append(tree)

    def _create_weighted_dataset(self, X, y, weights):
        ''' sample the dataset (with replacement) with weights[idx] being the probability of selecting the idx-th row '''
        X_w = np.empty((0, X.shape[1])) 
        y_w = np.empty((0,))
        rng = np.random.default_rng() # generate random number between 0 and 1
        
        for idx in range(X.shape[0]):
            rnd = rng.random()
            row_idx = self._get_sample_idx(rnd) 
            
            X_w = np.append(X_w, [X[row_idx]], axis=0) # add a selected row to X_w using weighted probabilities
            y_w = np.append(y_w, [y[row_idx]])

        return X_w, y_w
                    
    # rnd <- a random number between 0 and 1
    def _get_sample_idx(self, rnd):
        ''' get idx in X of sample whose range random number falls into '''
        
        for i in range(len(self.sample_ranges)):
            if self.sample_ranges[i] > rnd:
                return i
                
        return -1 # if failed return -1
            
    def predict(self, X):
        ''' predict values for a dataset '''
        X = np.array(X)
        y_preds = []
        
        for x in X:
            y_preds.append(self._predict_one(x))
            
        return y_preds

    def _predict_one(self, x):
        ''' predict value for one datapoint '''
        votes_weighted_by_say = [0] * self.n_classes

        # sum up the votes for each class, where each tree's vote is weighted as 1 * tree.say (tree.a_k)
        for k in range(self.k_classifiers):
            votes_weighted_by_say[int(self.trees[k].make_prediction(x, self.trees[k].root))] += self.say[k]

        return np.argmax(votes_weighted_by_say)