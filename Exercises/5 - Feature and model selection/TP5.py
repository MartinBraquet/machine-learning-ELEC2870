import numpy as np
from competitive_learning import get_inits, comp_learning, show_quantization
from linear_model import MyLinearRegressor, compute_rmse
import sklearn.feature_selection

# Data building
X = np.random.random_sample((100,6))

e = np.random.normal(0,0.01, (100,))
t = 2 * np.sin(2 * X[:,0]) * X[:,1] + 4 * (X[:,2] - 0.5)**2 + X[:,3] + e

# Features selection
corr = np.zeros((6,))
for i in range(6):
    corr[i] = np.corrcoef(X[:,i], t)[0,1]
    
mut = sklearn.feature_selection.mutual_info_regression(X, t)

X_full = X
X = X_full[:,:4]

X_train = X[:70,:]
X_val = X[:30,:]