# coding: utf-8

## Imports

from show_quantization import show_quantization
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


## Loading data

df = scipy.io.loadmat(f"data/diabetes.mat")
X = df['X']
t = df['t']
t = t[:,0]
title = ["age", "sex", "bmi", "blood_pressure", "serum_1", "serum_2", "serum_3", "serum_4", "serum_5", "serum_6"]

use_seaborn = True
plot_graphs = False

if plot_graphs:
    
    fig, axs = plt.subplots(nrows=4,ncols=3)
    
    if use_seaborn:
        #fig.suptitle('Just some random samples')
        
        for i in range (10):
            d = {title[i]: X[:,i], 't': t}
            Xpd = pd.DataFrame(data=d, columns=[title[i], "t"])
            sns.set() # Set seaborn defaults before call to figure
            
            # Showing the vector quantization
            sns.scatterplot(data=Xpd, x = title[i], y = "t", ax=axs[(i-i%3)//3][i%3])
    else:
        for i in range (10):
            a = (i-i%3)//3
            b = i%3
            
            axs[a,b].scatter(X[:,i],t)
            axs[a,b].set_title(title[i])
            axs[a,b].set_ylabel('t')
            axs[a,b].set_xlabel(title[i])
    
    plt.show()
    plt.close()

correlation = np.zeros(10,)
for i in range (10):
    res = np.corrcoef(X[:,i],t)
    correlation[i] = res[0,1]
print('Correlation: ', correlation)

ind_max_correl = np.argmax(correlation)
print('Index of max correlation: ', ind_max_correl)

X = np.transpose(X)

### Linear Regression

#%%
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
from math import sqrt

# Computes the Root Mean Square Error
def compute_rmse(predict, target):
    # Implement RMSE
    return sqrt(mean_squared_error(predict, target))

class MyLinearRegressor(BaseEstimator):
    def __init__(self, add_bias=True):
        super().__init__()
        self.add_bias = add_bias
        
    def fit(self, X, y):
        # TODO:
        # Compute the coefficients for the linear regression
        # DO NOT FORGET the add_bias argument
        if self.add_bias:
            X = np.vstack((np.ones(len(np.transpose(X))), X))
        self.w = np.linalg.inv(X @ np.transpose(X)) @ X @ y
        return self
    
    def predict(self, X):
        # TODO: 
        # Return y the solution based on the coefficients (computed beforehand)
        # applied on X
        if self.add_bias:
            X = np.vstack((np.ones(len(np.transpose(X))), X))
        y = self.w @ X
        return y
    
    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X)
    
    def score(self, X, y_true):
        y = self.predict(X)
        return compute_rmse(y, y_true)

### PCA (+lin.reg.)

#%%
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class MyStandardScaler():
    def __init__(self, with_std=True):
        self.with_std = with_std
    
    def fit(self, X, y=None):
        '''
            Computes the mean and standard deviation (if with_std is True)
            based on X
            Returns itself
        '''
        self.mean = np.mean(X, axis=0)
        if self.with_std:
            X = X - self.mean
            self.std = np.sqrt(np.var(X, axis=0))
        return self

    def transform(self, X):
        '''
            Transforms X based on the previously computed mean and std
            and returns it
        '''
        X = X - self.mean
        if self.with_std:
            X = X / self.std
        return X
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class MyPCA():
    def __init__(self, n_components, ):
        self.n_comps = n_components
        
    def fit(self, X, y=None):
        # TODO:
        # Implement PCA with SVD & save the coefficients
        return self
    
    def transform(self, X):
        # TODO:
        # Transform X by aggregating the features with the PCA
        # coefficients computed in 'fit'
        return None
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


n_comps = 2

# TODO: 
# Replace the default implementation with your own and CHECK that you obtain the same result
# pca = Pipeline([('scaler', StandardScaler(with_std=True)), ('pca', PCA(n_components=n_comps))])
# pca.fit_transform(X) # Here X contains the loaded data

MyLinearRegressor1 = MyLinearRegressor()
MyLinearRegressor1.fit(X=X, y=t)
score1 = MyLinearRegressor1.score(X=X, y_true=t)

print('Score 1: ', score1)

X1 = X[ind_max_correl,:]

MyLinearRegressor2 = MyLinearRegressor()
MyLinearRegressor2.fit(X=X1, y=t)
score2 = MyLinearRegressor2.score(X=X1, y_true=t)

print('Score 1: ', score2)

plt.scatter(X1,t)
x = [min(X1), max(X1)]
plt.plot(x,MyLinearRegressor2.predict(x))
plt.xlabel('X')
plt.ylabel('t')
plt.show()