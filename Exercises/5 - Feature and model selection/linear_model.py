import scipy.io
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.base import BaseEstimator

# Compute the Root Mean Square Error
def compute_rmse(predict, target):
    if len(target.shape) == 2:
        target = target.squeeze()
    if len(predict.shape) == 2:
        predict = predict.squeeze()
    diff = target - predict
    if len(diff.shape) == 1:
        diff = np.expand_dims(diff, axis=-1)
    rmse = np.sqrt(diff.T@diff / diff.shape[0])
    return float(rmse)

class MyLinearRegressor(BaseEstimator):
    def __init__(self, add_bias=True):
        super().__init__()
        self.add_bias = add_bias
        
    def fit(self, X, y):
        if self.add_bias:
            X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=-1)
        if len(y.shape) < 2:
            y = np.expand_dims(y, axis=-1)
        self.coeffs = np.linalg.inv(X.T @ X)@(X.T @ y)
        self.bias = self.coeffs[-1,0] if self.add_bias else 0
        self.coeffs = self.coeffs[:-1,:] if self.add_bias else self.coeffs
        return self
    
    def predict(self, X):
        y = X@self.coeffs + self.bias
        y = np.squeeze(y, axis=-1)
        return y
    
    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X)
    
    def score(self, X, y_true):
        y = self.predict(X)
        return compute_rmse(y, y_true)