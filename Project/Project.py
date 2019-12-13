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

save = 0

import numpy as np
import math
import matplotlib
import tikzplotlib
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns; sns.set()
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

    if save:
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
    
    if save:
        tikzplotlib.save("MI.tex")
    
    plt.show()


    return mut


# Linear regression
def project_linear_regression(X, Y):
    print('Linear regression...')

    model = LinearRegression()
    
    Y_pred = model.fit(X, Y).predict(X)
    
    RMSE = rmse(Y, Y_pred)
    print('Linear regression RMSE :', RMSE)

    error = math.sqrt(np.mean(bootstrap_point632_score(model, X, Y)))
    print('Linear regression bootstrap 632 error :', error)

    return error


# K-nearest neighbours
def project_KNN(X, Y, n, method_name='KNN', compute=True):
    print('KNN...')
    
    neighbors = np.arange(1, n + 1, 1)
    if compute:
        error_knn = np.zeros(n)
        for i in range(0, n):
            model = KNeighborsRegressor(n_neighbors=neighbors[i], metric='euclidean')
            Y_pred = model.fit(X, Y).predict(X)
            RMSE = rmse(Y, Y_pred)
            error_knn[i] = math.sqrt(np.mean(bootstrap_point632_score(model, X, Y, n_splits=100)))
            print(method_name, 'RMSE ( k =', neighbors[i], ') :', RMSE)
            print(method_name, 'bootstrap 632 error ( k =', neighbors[i], ') :', error_knn[i])
    else:
        if method_name == 'KNN':
            error_knn = np.array([44.63007435, 42.91722747, 42.42618258, 42.39780082, 42.54022997, 42.59120806, 42.71413344, 42.92639601, 42.86850688, 43.09182013, 43.36117329, 43.41121083, 43.65708279, 43.69896532, 43.86472366, 43.78765416, 44.00862374, 44.30227774, 44.29008714, 44.46076153, 44.53703636, 44.63881345, 44.69277326, 44.75460404, 44.81400926, 44.9321936, 45.14968511, 44.91520762, 45.18015732, 45.08772065, 45.30150145, 45.21632641, 45.3992413, 45.54305763, 45.44595713, 45.67029185, 45.71647618, 45.55042106, 45.79218583, 45.80604866, 45.89558866, 45.95351281, 45.95303281, 46.01579606, 46.04745272, 46.09775259, 46.17958316, 46.28358257, 46.26775877, 46.40855521])
        else:
            error_knn = np.array([52.73763501, 50.26260653, 49.43199863, 49.02281048, 48.68253806, 48.85401044, 48.67349653, 48.72226453, 48.57146228, 48.50207035, 48.56443761, 48.51289697, 48.53336497, 48.50314503, 48.47753564, 48.47326746, 48.57352092, 48.5564091, 48.4619637, 48.53501412, 48.50419066, 48.56703669, 48.5808846, 48.60146488, 48.61727791, 48.61611907, 48.75758009, 48.71311524, 48.77881461, 48.77234174, 48.8188231, 48.78412026, 48.7703691, 48.80950418, 48.83629899, 48.88289063, 48.93794985, 48.96546474, 49.01236155, 49.07860634, 48.99108467, 49.00323927, 48.94704306, 48.8915739, 49.09580907, 49.06468691, 49.03424426, 49.22457642, 49.13059562, 49.23870995])
    plt.plot(neighbors, error_knn, 'b')
    plt.xlabel('neighbours')
    plt.ylabel('Error' + method_name)

    if save:
        tikzplotlib.save("KNN.tex")
    
    plt.show(block=False)

    return error_knn

# Lasso (no feature selection)
def project_lasso(X, Y, n):
    print('Lasso...')
    
    alpha = np.logspace(-4.0, 4.0, num=n)
    
    '''
    error_lasso = np.zeros(n)
    for i in range(0, n):
        model = linear_model.Lasso(alpha=alpha[i])
        Y_pred = model.fit(X, Y).predict(X)
        RMSE = rmse(Y, Y_pred)
        error_lasso[i] = math.sqrt(np.mean(bootstrap_point632_score(model, X, Y, n_splits=100)))
        print('Lasso RMSE ( ', i, ') :', RMSE)
        print('Lasso bootstrap 632 error ( ', i, ') :', error_lasso[i])
    '''
    
    error_lasso = np.array([106.2375631, 106.16187842, 106.13303232, 106.26647534, 106.05651335, 106.20582387, 106.2186026, 106.08258511, 106.19265945, 106.29189452, 106.19592589, 106.1879064, 106.27420472, 106.024182, 106.21584605, 106.09416867, 106.07558031, 106.02752251, 106.09461169, 105.92143377, 105.82545833, 105.84943091, 105.60666613, 105.17078646, 104.7853639, 104.39823394, 103.74839552, 102.7847098, 102.004051, 100.31135177, 98.92064139, 96.73184041 , 93.78242505 , 90.13574035 , 85.84676636 , 81.98082039 , 81.43493206, 81.2762649, 81.41777602, 81.25663058, 81.36819753, 81.10169265, 81.36737431, 81.33121319, 81.51729483, 81.14111805, 81.43614077, 81.29413601, 81.21898883, 81.1998088])

    plt.plot(alpha, error_lasso, 'b')
    plt.xlabel('alpha')
    plt.ylabel('Error Lasso')
    plt.xscale('log')

    if save:
        tikzplotlib.save("Lasso.tex")
    
    plt.show(block=False)
    
    return error_lasso


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
pca = PCA(n_components=4)
X_PCA = pca.fit_transform(X_full.values)
print(pca.explained_variance_ratio_)


### Algorithms test

X = X_select

error_lin_reg = project_linear_regression(X, Y.values)

print(error_lin_reg)

n = 5

error_knn = project_KNN(X, Y.values, n)
error_knn_PCA = project_KNN(X_PCA, Y.values, n)

print(error_knn)
print(error_knn_PCA)

error_lasso = project_lasso(X, Y.values, n)

print(error_lasso)
