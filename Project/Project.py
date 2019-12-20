'''
LELEC2870 - Machine Learning: Project code
Date: 20/12/2019
Authors:
    - Martin Braquet
    - Amaury Gouverneur
    
Non standard package to install with pip:
    - torch
    - tikzplotlib (optional)
'''

save_mut_corr = False

compute_PCA = False
save_PCA = False

compute_bootstrap = False
save_bootstrap = False

compute_lasso = False
save_lasso = False

compute_RR = False
save_RR = False

compute_tree = False
save_tree = False

compute_adaboost_tree = False
save_adaboost_tree = False

compute_rand_tree = False
save_rand_tree = False

compute_boost_tree = False
save_boost_tree = False

compute_KNN = False
save_KNN = False

compute_MLP_neurons = False
compute_MLP_epochs = False
compute_MLP_best = False
save_MLP = False


import numpy as np
import math
import tikzplotlib
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from sklearn.metrics import mutual_info_score
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from mlxtend.evaluate import bootstrap_point632_score
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error


def load_data():
    def data_processing(Xpd):
        Xpd = time_to_coord(Xpd)
        Xpd = wd_to_coord(Xpd)
    
        X_full = Xpd
        X_full = X_full.drop(columns="wd")
        #X_full = X_full.drop(columns="WSPM")
        X_full = X_full.drop(columns="wdc")
        X_full = X_full.drop(columns="year")
        X_full = X_full.drop(columns="month")
        X_full = X_full.drop(columns="day")
        X_full = X_full.drop(columns="hour")
        # X_full = X_full.to_numpy()
    
        X_full_col = X_full.columns
        scaler = preprocessing.StandardScaler()
        X_full = scaler.fit_transform(X_full)  # Smaller error with scale instead of normalize
        X_full = pd.DataFrame(X_full)
        X_full.columns = X_full_col
        
        return X_full
    
    Xpd = pd.read_csv("Datasets/X1.csv", sep=',')
    
    Xpd2 = pd.read_csv("Datasets/X2.csv", sep=',')

    X_full = data_processing(Xpd)
    X_full2 = data_processing(Xpd2)

    Y = pd.read_csv("Datasets/Y1.csv", header=None, names=['PM2.5'])

    return Xpd, X_full, Y, X_full2


# New features: time, day time (Earth spin), year time (Earth rotation around Sun)
def time_to_coord(X1):
    time = np.zeros(X1.shape[0])
    theta_day = np.zeros(X1.shape[0])
    theta_year = np.zeros(X1.shape[0])
    for index, row in X1.iterrows():
        t = datetime(row['year'], row['month'], row['day'], row['hour'])
        time[index] = (t - datetime(2013, 1, 1)).total_seconds()
        time_year = (t - datetime(row['year'], 1, 1)).total_seconds()
        theta_year[index] = 2 * np.pi * time_year / (
                    datetime(row['year'] + 1, 1, 1) - datetime(row['year'], 1, 1)).total_seconds()
        theta_day[index] = 2 * np.pi * row['hour'] / 24

    cos_theta_year = np.cos(theta_year)
    sin_theta_year = np.sin(theta_year)
    cos_theta_day = np.cos(theta_day)
    sin_theta_day = np.sin(theta_day)
    X1.insert(X1.shape[1], "time", time, True)
    X1.insert(X1.shape[1], "syear", sin_theta_year, True)
    X1.insert(X1.shape[1], "cyear", cos_theta_year, True)
    X1.insert(X1.shape[1], "sday", sin_theta_day, True)
    X1.insert(X1.shape[1], "cday", cos_theta_day, True)
    return X1


# Polar to cartesian wind coordinates
def wd_to_coord(X1):
    for index, row in X1.iterrows():
        if row['wd'] == 'E':
            X1.at[index, 'wdc'] = 2 * np.pi * (0 / 16)
        elif row['wd'] == 'ENE':
            X1.at[index, 'wdc'] = 2 * np.pi * (1 / 16)
        elif row['wd'] == 'NE':
            X1.at[index, 'wdc'] = 2 * np.pi * (2 / 16)
        elif row['wd'] == 'NNE':
            X1.at[index, 'wdc'] = 2 * np.pi * (3 / 16)
        elif row['wd'] == 'N':
            X1.at[index, 'wdc'] = 2 * np.pi * (4 / 16)
        elif row['wd'] == 'NNW':
            X1.at[index, 'wdc'] = 2 * np.pi * (5 / 16)
        elif row['wd'] == 'NW':
            X1.at[index, 'wdc'] = 2 * np.pi * (6 / 16)
        elif row['wd'] == 'WNW':
            X1.at[index, 'wdc'] = 2 * np.pi * (7 / 16)
        elif row['wd'] == 'W':
            X1.at[index, 'wdc'] = 2 * np.pi * (8 / 16)
        elif row['wd'] == 'WSW':
            X1.at[index, 'wdc'] = 2 * np.pi * (9 / 16)
        elif row['wd'] == 'SW':
            X1.at[index, 'wdc'] = 2 * np.pi * (10 / 16)
        elif row['wd'] == 'SSW':
            X1.at[index, 'wdc'] = 2 * np.pi * (11 / 16)
        elif row['wd'] == 'S':
            X1.at[index, 'wdc'] = 2 * np.pi * (12 / 16)
        elif row['wd'] == 'SSE':
            X1.at[index, 'wdc'] = 2 * np.pi * (13 / 16)
        elif row['wd'] == 'SE':
            X1.at[index, 'wdc'] = 2 * np.pi * (14 / 16)
        else:
            X1.at[index, 'wdc'] = 2 * np.pi * (15 / 16)

    wd = X1.loc[:, ['wdc']].to_numpy()
    #wd_speed = X1.loc[:, ['WSPM']].to_numpy()
    sin_wd = np.sin(wd)# * wd_speed
    cos_wd = np.cos(wd)# * wd_speed
    X1.insert(13, "swd", sin_wd, True)
    X1.insert(14, "cwd", cos_wd, True)
    return X1


def rmse(predictions, targets):
    return math.sqrt(mean_squared_error(predictions, targets))


# Correlation
def project_correlation(X, Y, fig=True):
    print('Correlation...')

    XY = pd.concat([X, Y], axis=1)
    corr = XY.corr()

    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        sns.heatmap(np.abs(corr), mask=mask, annot=True, cmap=plt.cm.Reds, vmin=0, vmax=1, linewidths=.5).set_title('Correlation')

    if save_mut_corr:
        tikzplotlib.save("LaTeX/correlation.tex")
    
    plt.show()

    return corr


# Mutual information (buggy, not the same)
def project_mutual_info(X, Y):
    print('Mutual information...')

    XY = pd.concat([X, Y], axis=1)
    XY_col = XY.columns
    XY = XY.values
    n = XY.shape[1]
    mut = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mut[i, j] = mutual_info_score(XY[:, i], XY[:, j]) if j >= i else mut[j, i]

    diag_mut = np.copy(np.diag(mut))
    for i in range(n):
        for j in range(n):
            mut[i, j] = mut[i, j] / math.sqrt(diag_mut[i] * diag_mut[j])
    mut = pd.DataFrame(mut)
    mut.columns = XY_col
    mut.index = mut.columns

    mask = np.zeros_like(mut)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        sns.heatmap(np.abs(mut), mask=mask, annot=True, cmap=plt.cm.Reds, vmin=0, vmax=1, linewidths=.5).set_title('Mutual Information')
    
    if save_mut_corr:
        tikzplotlib.save("LaTeX/MI.tex")
    
    plt.show()


    return mut



#Error analysis for the Bootstrap on linear regression
def error_analysis_bootstrap(X, Y):
    
    n_splits = 10
    n = 1000
    
    error = np.zeros((n, n_splits))
    
    if compute_bootstrap:
        for i in range(n):
            error[i,:] = np.sqrt(bootstrap_point632_score(LinearRegression(), X, Y, n_splits=n_splits))
        np.save('numpy/error_bootstrap.npy', error)
    else:
        error = np.load('numpy/error_bootstrap.npy')
                
        
    print(np.std(np.mean(error,1)))
    print(np.std(np.reshape(error,(n*n_splits,1))[:n]))
    
    bins=np.arange(42,47,0.1)
    fig, ax = plt.subplots(1,1)
    ax.hist(np.mean(error,1), bins=bins, facecolor='blue', alpha=0.7, label='10 splits')
    ax.hist(np.reshape(error,(n*n_splits,1))[:n], bins=bins, facecolor='red', alpha=0.7, label='1 split')
    ax.set_xticks(bins[:-1])  
    plt.xticks(np.arange(42,47.4,0.5))
    plt.xlabel('Error [ug/m^3]')
    plt.ylabel('Samples')
    plt.legend()

    if save_bootstrap:
        tikzplotlib.save("LaTeX/error_analysis_bootstrap.tex")
    
    plt.show()


#PCA error for different numbers of principal components
def error_PCA(X, Y):
    
    n = 17
    
    PC = np.arange(1, n+1, 1)
    
    error = np.zeros((n,))
    
    if compute_PCA:
        for i in PC:
            X_PCA = PCA(n_components=i).fit_transform(X)
            error[i-1] = np.sqrt(np.mean(bootstrap_point632_score(LinearRegression(), X_PCA, Y, n_splits=50)))
        np.save('numpy/error_PCA_components.npy', error)
    else:
        error = np.load('numpy/error_PCA_components.npy')
    
    plt.scatter(PC, error)
    plt.xlabel('Number of components')
    plt.ylabel('Error [ug/m^3]')

    if save_PCA:
        tikzplotlib.save("LaTeX/error_PCA.tex")
    
    plt.show()

# Linear regression
def project_linear_regression(X, Y, method_name):
    print('Linear regression...')

    model = LinearRegression()
    
    Y_pred = model.fit(X, Y).predict(X)
    RMSE = rmse(Y, Y_pred)
    print('Linear regression RMSE', method_name, ' :', RMSE)

    error = math.sqrt(np.mean(bootstrap_point632_score(model, X, Y)))
    print('Linear regression bootstrap 632 error', method_name, ' :', error)

    return error


# Lasso (no feature selection)
def project_lasso(X_full_pd, X_select, X_PCA, Y):
    print('Lasso...')
    
    X_full = X_full_pd.values
    
    lambda_lasso = np.logspace(-2.0, 1.5, num=50)
        
    error_lasso_full = compute_error_lasso(X_full, Y, lambda_lasso, 'full')
    error_lasso_select = compute_error_lasso(X_select, Y,  lambda_lasso, 'select')
    error_lasso_PCA = compute_error_lasso(X_PCA, Y,  lambda_lasso, 'PCA')
    
    plt.xscale('log')
    plt.scatter(lambda_lasso, error_lasso_full, label='Full features')
    plt.scatter(lambda_lasso, error_lasso_select, label='Selected features')
    plt.scatter(lambda_lasso, error_lasso_PCA, label='PCA features')
    plt.xlabel('lambda')
    plt.ylabel('Error [ug/m^3]')
    plt.legend()

    if save_lasso:
        tikzplotlib.save("LaTeX/lasso.tex")
    
    plt.show()
    
    plot_weight_lasso(X_full_pd, Y)
    
def plot_weight_lasso(Xpd, Y):
    model = Lasso(alpha=0.1)
    model.fit(Xpd.values, Y)
    coef = pd.Series(model.coef_, index = Xpd.columns)
    
    imp_coef = pd.concat([coef.sort_values().head(10), coef.sort_values().tail(10)])
    matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
    imp_coef.plot(kind = "barh")
    plt.xlabel('Weight')
    
    if save_lasso:
        tikzplotlib.save("LaTeX/lasso_weights.tex")
    
    plt.show()
    
    
def compute_error_lasso(X, Y, lambda_lasso, method_name):
    
    if compute_lasso:
        
        error_lasso = np.zeros(lambda_lasso.shape)
        for i in range(0, 50):
            model = Lasso(alpha=lambda_lasso[i])
            Y_pred = model.fit(X, Y).predict(X)
            RMSE = rmse(Y, Y_pred)
            error_lasso[i] = RMSE
            print(method_name, 'RMSE ( lambda =', lambda_lasso[i], ') :', RMSE)
            error_lasso[i] = math.sqrt(np.mean(bootstrap_point632_score(model, X, Y.ravel(), n_splits=10)))
            print(method_name, 'bootstrap 632 error ( lambda =', lambda_lasso[i], ') :', error_lasso[i])   
        np.save('numpy/error_lasso_{}.npy'.format(method_name), error_lasso)
    else:
        error_lasso = np.load('numpy/error_lasso_{}.npy'.format(method_name))

    return error_lasso


# Ridge regression
def project_ridge_regression(X_full, X_select, X_PCA, Y):
    print('Ridge regression...')
    
    error_RR_full = compute_error_RR(X_full, Y, 'full')
    error_RR_select = compute_error_RR(X_select, Y, 'select')
    error_RR_PCA = compute_error_RR(X_PCA, Y, 'PCA')
    
    alpha_RR = np.logspace(-2.0, 4.0, num=50)
    plt.xscale('log')
    plt.scatter(alpha_RR, error_RR_full, label='Full features')
    plt.scatter(alpha_RR, error_RR_select, label='Selected features')
    plt.scatter(alpha_RR, error_RR_PCA, label='PCA features')
    plt.xlabel('lambda')
    plt.ylabel('Error [ug/m^3]')
    plt.legend()

    if save_RR:
        tikzplotlib.save("LaTeX/RR.tex")
    
    plt.show()
    
def compute_error_RR(X, Y, method_name):
    
    alpha_RR = np.logspace(-2.0, 4.0, num=50)
    if compute_RR:
        error_RR = np.zeros(50)
        for i in range(0, 50):
            model = Ridge(alpha = alpha_RR[i])
            Y_pred = model.fit(X, Y).predict(X)
            RMSE = rmse(Y, Y_pred)
            error_RR[i] = math.sqrt(np.mean(bootstrap_point632_score(model, X, Y, n_splits=100)))
            print(method_name, 'RMSE ( k =', alpha_RR[i], ') :', RMSE)
            print(method_name, 'bootstrap 632 error ( k =', alpha_RR[i], ') :', error_RR[i])   
        np.save('numpy/error_RR_{}.npy'.format(method_name), error_RR)
    else:
        error_RR = np.load('numpy/error_RR_{}.npy'.format(method_name))

    return error_RR


# Tree regression
def project_tree_regression(X_full, X_select, X_PCA, Y, n):
    print('Tree regression...')
    
    error_tree_full = compute_error_tree(X_full, Y, 'full', n)
    error_tree_select = compute_error_tree(X_select, Y, 'select', n)
    error_tree_PCA = compute_error_tree(X_PCA, Y, 'PCA', n)
    
    depth = np.arange(1, n + 1, 1)
    
    
    plt.scatter(depth, error_tree_full, label='Full features')
    plt.scatter(depth, error_tree_select, label='Selected features')
    plt.scatter(depth, error_tree_PCA, label='PCA features')
    plt.xlabel('Max depth')
    plt.ylabel('Error [ug/m^3]')
    plt.legend()

    if save_tree:
        tikzplotlib.save("LaTeX/tree.tex")
    
    plt.show()
    
def compute_error_tree(X, Y, method_name, n):
    
    depth = np.arange(1, n + 1, 1)
    if compute_tree:
        error_tree = np.zeros(n)
        for i in range(0, n):
            model = DecisionTreeRegressor(max_depth = depth[i])
            Y_pred = model.fit(X, Y).predict(X)
            RMSE = rmse(Y, Y_pred)
            error_tree[i] = RMSE
            error_tree[i] = math.sqrt(np.mean(bootstrap_point632_score(model, X, Y.ravel(), n_splits=10)))
            print(method_name, 'RMSE ( k =', depth[i], ') :', RMSE)
            print(method_name, 'bootstrap 632 error ( k =', depth[i], ') :',  error_tree[i])   
        np.save('numpy/error_tree_{}.npy'.format(method_name), error_tree)
    else:
        error_tree = np.load('numpy/error_tree_{}.npy'.format(method_name))

    return error_tree


# Tree regression
def project_adaboost_tree_regression(X_full, X_select, X_PCA, Y, n):
    print('Adaboost tree regression...')
    
    error_tree_full = compute_error_adaboost_tree(X_full, Y, 'full', n)
    error_tree_select = compute_error_adaboost_tree(X_select, Y, 'select', n)
    error_tree_PCA = compute_error_adaboost_tree(X_PCA, Y, 'PCA', n)
    
    depth = np.arange(1, n + 1, 1)
    
    
    plt.scatter(depth, error_tree_full, label='Full features')
    plt.scatter(depth, error_tree_select, label='Selected features')
    plt.scatter(depth, error_tree_PCA, label='PCA features')
    plt.xlabel('Max depth')
    plt.ylabel('Error [ug/m^3]')
    plt.legend()

    if save_adaboost_tree:
        tikzplotlib.save("LaTeX/adaboost_tree.tex")
    
    plt.show()
    
def compute_error_adaboost_tree(X, Y, method_name, n):
    
    depth = np.arange(1, n + 1, 1)
    if compute_adaboost_tree:
        error_tree = np.zeros(n)
        for i in range(0, n):
            model = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth = depth[i]))
            Y_pred = model.fit(X, Y).predict(X)
            RMSE = rmse(Y, Y_pred)
            error_tree[i] = RMSE
            error_tree[i] = math.sqrt(np.mean(bootstrap_point632_score(model, X, Y.ravel(), n_splits=10)))
            print(method_name, 'RMSE ( k =', depth[i], ') :', RMSE)
            print(method_name, 'bootstrap 632 error ( k =', depth[i], ') :',  error_tree[i])   
        np.save('numpy/error_adaboost_tree_{}.npy'.format(method_name), error_tree)
    else:
        error_tree = np.load('numpy/error_adaboost_tree_{}.npy'.format(method_name))

    return error_tree

# Tree regression
def project_random_tree_regression(X_full, X_select, X_PCA, Y, n):
    print('Random tree regression...')
    
    error_tree_full = compute_error_random_tree(X_full, Y, 'full', n)
    error_tree_select = compute_error_random_tree(X_select, Y, 'select', n)
    error_tree_PCA = compute_error_random_tree(X_PCA, Y, 'PCA', n)
    
    depth = np.arange(1, n + 1, 1)
    
    
    plt.scatter(depth, error_tree_full, label='Full features')
    plt.scatter(depth, error_tree_select, label='Selected features')
    plt.scatter(depth, error_tree_PCA, label='PCA features')
    plt.xlabel('Max depth')
    plt.ylabel('Error [ug/m^3]')
    plt.legend()

    if save_rand_tree:
        tikzplotlib.save("LaTeX/random_tree.tex")
    
    plt.show()
    
def compute_error_random_tree(X, Y, method_name, n):
    
    depth = np.arange(1, n + 1, 1)
    if compute_rand_tree:
        error_tree = np.zeros(n)
        for i in range(0, n):
            model = RandomForestRegressor(max_depth = depth[i])
            Y_pred = model.fit(X, Y).predict(X)
            RMSE = rmse(Y, Y_pred)
            error_tree[i] = RMSE
            error_tree[i] = math.sqrt(np.mean(bootstrap_point632_score(model, X, Y.ravel(), n_splits=10)))
            print(method_name, 'RMSE ( k =', depth[i], ') :', RMSE)
            print(method_name, 'bootstrap 632 error ( k =', depth[i], ') :',  error_tree[i])   
        np.save('numpy/error_random_tree_{}.npy'.format(method_name), error_tree)
    else:
        error_tree = np.load('numpy/error_random_tree_{}.npy'.format(method_name))

    return error_tree


# Boosting tree regression
def project_boost_tree_regression(X_full, X_select, X_PCA, Y, n):
    print('Boosting tree regression...')
    
    error_tree_full = compute_error_boost_tree(X_full, Y, 'full', n)
    error_tree_select = compute_error_boost_tree(X_select, Y, 'select', n)
    error_tree_PCA = compute_error_boost_tree(X_PCA, Y, 'PCA', n)
    
    depth = np.arange(1, n + 1, 1)
    
    
    plt.scatter(depth, error_tree_full, label='Full features')
    plt.scatter(depth, error_tree_select, label='Selected features')
    plt.scatter(depth, error_tree_PCA, label='PCA features')
    plt.xlabel('Max depth')
    plt.ylabel('Error [ug/m^3]')
    plt.legend()

    if save_boost_tree:
        tikzplotlib.save("LaTeX/boost_tree.tex")
    
    plt.show()
    
def compute_error_boost_tree(X, Y, method_name, n):
    
    depth = np.arange(1, n + 1, 1)
    if compute_boost_tree:
        error_tree = np.zeros(n)
        for i in range(0, n):
            model = BaggingRegressor(base_estimator=DecisionTreeRegressor(max_depth = depth[i]))
            Y_pred = model.fit(X, Y).predict(X)
            RMSE = rmse(Y, Y_pred)
            error_tree[i] = RMSE
            error_tree[i] = math.sqrt(np.mean(bootstrap_point632_score(model, X, Y.ravel(), n_splits=10)))
            print(method_name, 'RMSE ( k =', depth[i], ') :', RMSE)
            print(method_name, 'bootstrap 632 error ( k =', depth[i], ') :',  error_tree[i])   
        np.save('numpy/error_boost_tree_{}.npy'.format(method_name), error_tree)
    else:
        error_tree = np.load('numpy/error_boost_tree_{}.npy'.format(method_name))

    return error_tree


# K-nearest neighbours
def project_KNN(X_full, X_select, X_PCA, Y, n):
    print('KNN...')
    
    error_knn_full = compute_error_KNN(X_full, Y, n, 'full')
    error_knn_select = compute_error_KNN(X_select, Y, n, 'select')
    error_knn_PCA = compute_error_KNN(X_PCA, Y, n, 'PCA')
    
    neighbors = np.arange(1, n + 1, 1)
    
    plt.scatter(neighbors, error_knn_full, label='Full features')
    plt.scatter(neighbors, error_knn_select, label='Selected features')
    plt.scatter(neighbors, error_knn_PCA, label='PCA features')
    plt.xlabel('Neighbours')
    plt.ylabel('Error [ug/m^3]')
    plt.legend()

    if save_KNN:
        tikzplotlib.save("LaTeX/KNN.tex")
    
    plt.show()
    
def compute_error_KNN(X, Y, n, method_name):
    
    neighbors = np.arange(1, n + 1, 1)
    if compute_KNN:
        error_knn = np.zeros(n)
        for i in range(0, n):
            model = KNeighborsRegressor(n_neighbors=neighbors[i], metric='euclidean')
            Y_pred = model.fit(X, Y).predict(X)
            RMSE = rmse(Y, Y_pred)
            error_knn[i] = math.sqrt(np.mean(bootstrap_point632_score(model, X, Y, n_splits=100)))
            print(method_name, 'RMSE ( k =', neighbors[i], ') :', RMSE)
            print(method_name, 'bootstrap 632 error ( k =', neighbors[i], ') :', error_knn[i])   
        np.save('numpy/error_knn_{}.npy'.format(method_name), error_knn)
    else:
        error_knn = np.load('numpy/error_knn_{}.npy'.format(method_name))

    return error_knn

# Multi-Layer Perceptron
def project_MLP(X_full, X_select, X_PCA, Y):
    
    def param_error_MLP(X_full, X_select, X_PCA, Y, n, param_var):
        
        error_MLP_full = compute_error_MLP(X_full, Y, n, 'full', param_var)
        if param_var != 'layers':
            error_MLP_select = compute_error_MLP(X_select, Y, n, 'select', param_var)
            error_MLP_PCA = compute_error_MLP(X_PCA, Y, n, 'PCA', param_var)
        
        x = np.arange(1, n, 1)
        
        plt.scatter(x, error_MLP_full, label='Full features')
        if param_var != 'layers':
            plt.scatter(x, error_MLP_select, label='Selected features')
            plt.scatter(x, error_MLP_PCA, label='PCA features')
        plt.xlabel(param_var)
        plt.ylabel('Error [ug/m^3]')
        plt.legend()
    
        if save_MLP:
            tikzplotlib.save('LaTeX/MLP_{}.tex'.format(param_var))
        
        plt.show()
        
    
    print('MLP...')
        
    param_error_MLP(X_full, X_select, X_PCA, Y, 50, 'neurons')
    param_error_MLP(X_full, X_select, X_PCA, Y, 81, 'epochs')
    param_error_MLP(X_full, X_select, X_PCA, Y, 21, 'layers')
    
    
    #error_MLP_best = compute_error_MLP(X_full, Y, 0, 'full', 'solo')

    
def compute_error_MLP(X, Y, n, method_name, param_var):

    if param_var == 'neurons':
        if compute_MLP_neurons:
            error_mlp = np.zeros(n)
            for i in range(0, n):
                h_sizes = [X.shape[1]] + ([i+1] * 8) + [1]
                model = MLP_model(h_sizes, 50, '')
                #model.fit(X, Y)
                #Y_pred = model.predict(X)
                #RMSE = rmse(Y, Y_pred)
                #print(method_name, 'RMSE ( neurons =', i, ') :', RMSE)
                error_mlp[i] = math.sqrt(np.mean(bootstrap_point632_score(model, X, Y, n_splits=10)))
                print(method_name, 'bootstrap 632 error (', param_var, '=', i+1, ') :', error_mlp[i])   
            np.save('numpy/error_mlp_{}_{}.npy'.format(method_name, param_var), error_mlp)
        else:
            error_mlp = np.load('numpy/error_mlp_{}_{}.npy'.format(method_name, param_var))

    elif param_var == 'epochs':
        if compute_MLP_epochs:
            error_mlp = np.zeros(n)
            for i in range(0, n):
                h_sizes = [X.shape[1]] + ([50] * 8) + [1]
                model = MLP_model(h_sizes, i+1, '')
                #model.fit(X, Y)
                #Y_pred = model.predict(X)
                #RMSE = rmse(Y, Y_pred)
                #print(method_name, 'RMSE ( neurons =', i, ') :', RMSE)
                error_mlp[i] = math.sqrt(np.mean(bootstrap_point632_score(model, X, Y, n_splits=10)))
                print(method_name, 'bootstrap 632 error (', param_var, '=', i+1, ') :', error_mlp[i])   
            np.save('numpy/error_mlp_{}_{}.npy'.format(method_name, param_var), error_mlp)
        else:
            error_mlp = np.load('numpy/error_mlp_{}_{}.npy'.format(method_name, param_var))
    elif param_var == 'layers':
        if compute_MLP_best:
            error_mlp = np.zeros(n)
            for i in range(0, n):
                h_sizes = [X.shape[1]] + ([50] * (i+1)) + [1]
                model = MLP_model(h_sizes, 10*(i+1), '')
                #model.fit(X, Y)
                #Y_pred = model.predict(X)
                #RMSE = rmse(Y, Y_pred)
                #print(method_name, 'RMSE ( neurons =', i, ') :', RMSE)
                error_mlp[i] = math.sqrt(np.mean(bootstrap_point632_score(model, X, Y, n_splits=5)))
                print(method_name, 'bootstrap 632 error (', param_var, '=', i+1, ') :', error_mlp[i])   
            np.save('numpy/error_mlp_{}_{}.npy'.format(method_name, param_var), error_mlp)
        else:
            error_mlp = np.load('numpy/error_mlp_{}_{}.npy'.format(method_name, param_var))
    elif param_var == 'solo':
        h_sizes = [X.shape[1]] + ([200] * 20) + [1]
        model = MLP_model(h_sizes, 300, '')
        #model.fit(X, Y)
        #Y_pred = model.predict(X)
        #RMSE = rmse(Y, Y_pred)
        #print(method_name, 'RMSE ( neurons =', i, ') :', RMSE)
        error_mlp = np.sqrt(bootstrap_point632_score(model, X, Y, n_splits=1))
        error_mlp_mean = np.mean(error_mlp)
        print('Solo MLP :', error_mlp)
        print('Solo MLP :', error_mlp_mean)
    else:
        print('Wrong param_var!')

    return error_mlp

class MLP(nn.Module):
    def __init__(self, h_sizes):
        super().__init__()

        self.h_sizes = h_sizes
        self.hidden = nn.ModuleList()
        for k in range(len(h_sizes)-1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k+1]))
            
        self.batch_hid = nn.BatchNorm1d(num_features=h_sizes[1])
                
        
    def forward(self, x):            
        for i, l in enumerate(self.hidden):
            x = self.hidden[i](x)
            if i != len(self.h_sizes)-2:
                x = F.relu(self.batch_hid(x))
        return x
    
class MLP_Dataset(Dataset):
    def __init__(self, data):
        self.x = np.transpose(torch.from_numpy(data[0])).float()
        self.y = torch.from_numpy(data[1]).float()
         
    def __getitem__(self, index): 
        return (self.x[:,index], self.y[index])
 
    def __len__(self):
        return len(self.y)
        
class MLP_model():
    
    _estimator_type = "regressor"
    
    
    def __init__(self, h_sizes, epochs, path):
        self.model = MLP(h_sizes)
        self.path = 'pytorch/MLP.pt'
        self.epochs = epochs
        self.h_sizes = h_sizes

    def fit(self, x, y):
        model = MLP(self.h_sizes)
        self.model = model
        
        lr = 0.01
        batch_size = 128
        epochs = self.epochs
    
        # Object where the data should be moved to:
        #   either the CPU memory (aka RAM + cache)
        #   or GPU memory
        #device = torch.device('cuda') # Note: cuda is the name of the technology inside NVIDIA graphic cards
        #model = model.to(device) # Transfer Network on graphic card.
        
        #torch.save(network.state_dict(), new_model_path)
       
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001) # regularization done with weight_decay
        lMSE = nn.MSELoss()
        
        train_set = MLP_Dataset((x, y))
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
        
        for epoch in range(epochs):
            
            model.train() # Set the network in training mode => weights become trainable aka modifiable
            for batch in train_loader:
                
                (x, y) = batch
                
                #print(x.shape)
    
                y_pred = model(x)  # Infer a batch through the network
                
                loss = lMSE(y_pred, y)
                
                #print('Loss: ', math.sqrt(loss.item()))
                
                optimizer.zero_grad()  # (Re)Set all the gradients to zero
                loss.backward()  # Compute the backward pass based on the gradients and activations
                optimizer.step()  # Update the weights
        
    def predict(self, x):
        return self.model(torch.from_numpy(x).float()).data.numpy()
    
    def get_params(self, deep=True):
        return {"h_sizes": self.h_sizes, "path": self.path, "epochs": self.epochs}
    
    def set_params(self, **parameters):
        self.load_state_dict(torch.load(self.path))
        return self



# Output prediction
def prediction(X, Y, X2):
    model = BaggingRegressor(base_estimator=DecisionTreeRegressor(max_depth = 33))
    Y2_pred = model.fit(X, Y).predict(X2)
    Y1_pred = model.predict(X)
    print(rmse(Y1_pred, Y))
    
    np.savetxt('Datasets/Y2_pred.csv', np.transpose(Y2_pred).ravel(), delimiter=',')
    
    return Y2_pred


# Data building
if 'Xpd' not in locals():
    print('Loading...')
    Xpd, X_full, Y, X2 = load_data()

# print(np.max(X_full,1))
# print(np.min(X_full,1))
nd, nf = X_full.shape  # nd: nbr of data, nf: nbr of features

### Features selection

if 'corr' not in locals():
    corr = project_correlation(X_full, Y)

if 'mut' not in locals():
    mut = project_mutual_info(X_full, Y)

if 'index' not in locals():
    index = mut.values[-1,:-1] > 0.2
    index[4] = False
    index[5] = False
    index[12] = False
    X_select = X_full.values[:,index]
    X2_select = X2.values[:,index]


'''
Keeps only the relevant features
 X_full - 0: SO2
          1: NO2
          2: CO
          3: O3
          4: TEMP
          5: PRES
          6: DEWP
          7: RAIN
          8: swd (sine of wind coordinate)
          9: cwd (cosine of wind coordinate)
          10: WSPM
          11: station
          12: time (time coordinate)
          13: syear (sine of time expressed around a year)
          14: cyear (cosine of time expressed around a year)
          15: sday (sine of time expressed around a day)
          16: cday (cosine of time expressed around a day)
'''

### Features extraction

# PCA
if 'X_PCA' not in locals():
    pca = PCA(n_components=3)
    X_PCA = pca.fit_transform(X_full.values)
    print(pca.explained_variance_ratio_)


# Error analysis
error_analysis_bootstrap(X_select, Y.values)

error_PCA(X_full, Y.values)



### Algorithms test

# Linear regression
error_lin_reg_full = project_linear_regression(X_full.values, Y.values, 'full')
error_lin_reg_select = project_linear_regression(X_select, Y.values, 'select')
error_lin_reg_PCA = project_linear_regression(X_PCA, Y.values, 'PCA')

# Ridge regression
project_ridge_regression(X_full.values, X_select, X_PCA, Y.values)

# Lasso
project_lasso(X_full, X_select, X_PCA, Y.values)

# KNN
project_KNN(X_full.values, X_select, X_PCA, Y.values, 50)

# Tree regression
project_tree_regression(X_full.values, X_select, X_PCA, Y.values,20)

# Adaboost tree regression
project_adaboost_tree_regression(X_full.values, X_select, X_PCA, Y.values,20)

# Random tree regression
project_random_tree_regression(X_full.values, X_select, X_PCA, Y.values,20)

# Boost tree regression
project_boost_tree_regression(X_full, X_select, X_PCA, Y, 40)

# MLP
project_MLP(X_full.values, X_select, X_PCA, Y.values)


### Prediction

Y2_pred = prediction(X_select, Y.values, X2_select)
