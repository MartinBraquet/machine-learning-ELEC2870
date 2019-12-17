'''
Questions: 
- Validation set for feature selection? No
- How to characterize wd and time? OK, sin/cos
- Normalization, how it works? standard scaler
- MI, how it works? Remove random effect
- Compute error on training or val set? K-fold cross validation (divide in K sets and repeat) or bootstrap 632 (better)
- Feature extraction: PCA,...

Greed search inside a K fold
'''

save_mut_corr = False

compute_lasso = True
save_lasso = False

compute_KNN = False
save_KNN = False

compute_MLP = False
save_MLP = True


import numpy as np
import math
import matplotlib
import tikzplotlib
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns#; sns.set()
from sklearn.metrics import mutual_info_score
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from mlxtend.evaluate import bootstrap_point632_score
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset


def load_data():
    Xpd = pd.read_csv("Datasets/X1.csv", sep=',')
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

    Y = pd.read_csv("Datasets/Y1.csv", header=None, names=['PM2.5'])

    return Xpd, X_full, Y


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
    return np.sqrt(((predictions - targets) ** 2).mean())


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
        tikzplotlib.save("correlation.tex")
    
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
        tikzplotlib.save("MI.tex")
    
    plt.show()


    return mut


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
        tikzplotlib.save("KNN.tex")
    
    plt.show(block=False)
    
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

'''
# Lasso (no feature selection)
def project_lasso(X_full, X_select, X_PCA, Y):
    print('Lasso...')
    
    lambda_lasso = np.logspace(-4.0, 4.0, num=50)
        
    error_lasso_full = compute_error_lasso(X_full, Y, lambda_lasso, 'full')
    error_lasso_select = compute_error_lasso(X_select, Y,  lambda_lasso, 'select')
    error_lasso_PCA = compute_error_lasso(X_PCA, Y,  lambda_lasso, 'PCA')
    
    plt.scatter(lambda_lasso, error_lasso_full, label='Full features')
    plt.scatter(lambda_lasso, error_lasso_select, label='Selected features')
    plt.scatter(lambda_lasso, error_lasso_PCA, label='PCA features')
    plt.xlabel('lambda')
    plt.ylabel('Error [ug/m^3]')
    plt.legend()

    if save_lasso:
        tikzplotlib.save("lasso.tex")
    
    plt.show(block=False)
    
def compute_error_lasso(X, Y, lambda_lasso, method_name):
    
    X = X[:3,:]
    Y = Y[:3]
    
    if compute_lasso:
        error_lasso = np.zeros(lambda_lasso.shape)
        for lambda_i in lambda_lasso:
            model = linear_model.Lasso(alpha=lambda_i)
            Y_pred = model.fit(X, Y).predict(X)
            RMSE = rmse(Y, Y_pred)
            print(method_name, 'RMSE ( lambda =', lambda_i, ') :', RMSE)
            error_lasso[i] = math.sqrt(np.mean(bootstrap_point632_score(model, X, Y, n_splits=100)))
            print(method_name, 'bootstrap 632 error ( lambda =', lambda_i, ') :', error_lasso[i])   
        np.save('numpy/error_lasso_{}.npy'.format(method_name), error_lasso)
    else:
        error_lasso = np.load('numpy/error_lasso_{}.npy'.format(method_name))

    return error_lasso
'''

# Multi-Layer Perceptron
def project_MLP(X_full, X_select, X_PCA, Y):
    print('MLP...')
    
    n = 50
    
    error_MLP_full = compute_error_MLP(X_full, Y, n, 'full')
    error_MLP_select = compute_error_MLP(X_select, Y, n, 'select')
    error_MLP_PCA = compute_error_MLP(X_PCA, Y, n, 'PCA')
    
    neurons = np.arange(1, n, 1)
    
    plt.scatter(neurons, error_MLP_full, label='Full features')
    plt.scatter(neurons, error_MLP_select, label='Selected features')
    plt.scatter(neurons, error_MLP_PCA, label='PCA features')
    plt.xlabel('Neurons per layer')
    plt.ylabel('Error [ug/m^3]')
    plt.legend()

    if save_MLP:
        tikzplotlib.save("MLP.tex")
    
    plt.show(block=False)
    
def compute_error_MLP(X, Y, n, method_name):
    class MLP(nn.Module):
        def __init__(self, nin, n):
            super().__init__()
    
            self.batch = nn.BatchNorm1d(num_features=n)
            
            self.lin1 = nn.Linear(nin, n)
            self.lin2 = nn.Linear(n, n)
            self.lin3 = nn.Linear(n, n)
            self.lin4 = nn.Linear(n, n)
            self.lin5 = nn.Linear(n, n)
            self.lin6 = nn.Linear(n, n)
            self.lin7 = nn.Linear(n, n)
            self.lin8 = nn.Linear(n, 1)
            
    
        def forward(self, x):
            
            x = F.relu(self.batch(self.lin1(x)))
            x = F.relu(self.batch(self.lin2(x)))
            x = F.relu(self.batch(self.lin3(x)))
            x = F.relu(self.batch(self.lin4(x)))
            x = F.relu(self.batch(self.lin5(x)))
            x = F.relu(self.batch(self.lin6(x)))
            x = F.relu(self.batch(self.lin7(x)))
            x = self.lin8(x)
            
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
        
        def __init__(self, nin, n, path):
            self.model = MLP(nin, n)
            self.nin = nin
            self.n = n
            self.path = 'pytorch/MLP.pt'
    
        def fit(self, x, y):
            model = self.model
            
            lr = 0.01
            batch_size = 128
            epochs = 50
        
            # Object where the data should be moved to:
            #   either the CPU memory (aka RAM + cache)
            #   or GPU memory
            #device = torch.device('cuda') # Note: cuda is the name of the technology inside NVIDIA graphic cards
            #network = DeepNetwork().to(device) # Transfer Network on graphic card.
            
            #torch.save(network.state_dict(), new_model_path)
           
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001) # regularization done with weight_decay
            lMSE = nn.MSELoss()
            
            train_set = MLP_Dataset((x, y))
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
            
            for epoch in range(epochs):
                i = 0
                
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
            return {"nin": self.nin, "n": self.n, "path": self.path}
        
        def set_params(self, **parameters):
            self.load_state_dict(torch.load(self.path))
            return self
            


    if compute_MLP:
        error_mlp = np.zeros(n)
        for i in range(0, n):
            model = MLP_model(X.shape[1], i+1, '')
            #model.fit(X, Y)
            #Y_pred = model.predict(X)
            #RMSE = rmse(Y, Y_pred)
            #print(method_name, 'RMSE ( neurons =', i, ') :', RMSE)
            error_mlp[i] = math.sqrt(np.mean(bootstrap_point632_score(model, X, Y, n_splits=10)))
            print(method_name, 'bootstrap 632 error ( neurons =', i+1, ') :', error_mlp[i])   
        np.save('numpy/error_mlp_{}.npy'.format(method_name), error_mlp)
    else:
        error_mlp = np.load('numpy/error_mlp_{}.npy'.format(method_name))

    return error_mlp


# Data building
if 'Xpd' not in locals():
    print('Loading...')
    Xpd, X_full, Y = load_data()

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
          8: WSPM
          9: station
          10: z (first time coordinate)
          11: sin_theta (second time coordinate)
          12: cos_theta (third time coordinate)
          12: wd_1 (first wind coordinate)
          13: wd_2 (second wind coordinate)
'''

### Features extraction

# PCA
if 'X_PCA' not in locals():
    pca = PCA(n_components=7)
    X_PCA = pca.fit_transform(X_full.values)
    print(pca.explained_variance_ratio_)


### Algorithms test
'''
# Linear regression
error_lin_reg_full = project_linear_regression(X_full.values, Y.values, 'full')


error_lin_reg_select = project_linear_regression(X_select, Y.values, 'select')
error_lin_reg_PCA = project_linear_regression(X_PCA, Y.values, 'PCA')

# KNN
n = 50
project_KNN(X_full.values, X_select, X_PCA, Y.values, n)
'''

# Lasso
#project_lasso(X_full.values, X_select, X_PCA, Y.values)

# MLP
project_MLP(X_full.values, X_select, X_PCA, Y.values)
