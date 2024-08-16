import numpy as np
import pandas as pd

class LinearRegression:
    ''' simple univariate linear regression using gradient descent for optimal fit '''

    def fit(self, X, y, iters=10, learning_rate=0.1):
        # this should set the final w (slope) and b (bias/y-intercept) values to be used when predicting

        # w and b start at 0
        w = b = 0
        dataset = np.c_[X, y]
        
        # for i iterations:
        # 1 - calculate the sum of the gradients calculated for all the training datapoints
        # 2 - update weight and bias by scaling sum by learning rate
        for i in range(iters):
            w, b = self._gradient(w, b, dataset) # 1
            w -= w * learning_rate
            b -= b * learning_rate
            
        self.w = w
        self.b = b

    def _gradient(self, w, b, dataset):
        # calculate the average of the gradients (of w and b) calculated for all the datapoints 
        N = dataset.shape[0] # number of datapoints
        w_grad_sum = b_grad_sum = 0
        
        for sample in dataset:
            x_i, y_i = sample[0], sample[1]
            
            w_grad_sum += -2 * x_i * (y_i - (w * x_i + b))
            b_grad_sum += -2 * (y_i - (w * x_i + b))

        return w_grad_sum/N, b_grad_sum/N
            
    def predict(self, X):
        pass

