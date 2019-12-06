'''
Questions: 
- Validation set for feature selection? No, 
- How to characterize wd and time? OK
- Normalization, how it works? standard scaler, 
- MI, how it works?
- Compute error on training or val set? Divide in K sets and repeat (bootstrap 632)
- Feature extraction: PCA,...

Greed search inside a K fold
'''

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

def load_data():
    Xpd = pd.read_csv("Datasets/X1.csv",sep = ',')
    Xpd = time_to_coord(Xpd)
    Xpd = wd_to_coord(Xpd)

    X_full = Xpd
    X_full = X_full.drop(columns="wd")
    X_full = X_full.drop(columns="WSPM")
    X_full = X_full.drop(columns="wdc")
    X_full = X_full.drop(columns="year")
    X_full = X_full.drop(columns="month")
    X_full = X_full.drop(columns="day")
    X_full = X_full.drop(columns="hour")
    #X_full = X_full.to_numpy()
    
    X_full_col = X_full.columns
    scaler = preprocessing.StandardScaler()
    X_full = scaler.fit_transform(X_full) # Smaller error with scale instead of normalize
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
        time[index] = (t-datetime(2013,1,1)).total_seconds()
        time_year = (t-datetime(row['year'],1,1)).total_seconds()
        theta_year[index] = 2 * np.pi * time_year / (datetime(row['year']+1,1,1)-datetime(row['year'],1,1)).total_seconds()
        theta_day[index] = 2 * np.pi * row['hour'] / 24
        
    cos_theta_year = np.cos(theta_year)
    sin_theta_year = np.sin(theta_year)
    cos_theta_day = np.cos(theta_day)
    sin_theta_day = np.sin(theta_day)
    X1.insert(X1.shape[1], "time", time, True)
    X1.insert(X1.shape[1], "sin_theta_year", sin_theta_year, True)
    X1.insert(X1.shape[1], "cos_theta_year", cos_theta_year, True)
    X1.insert(X1.shape[1], "sin_theta_day", sin_theta_day, True)
    X1.insert(X1.shape[1], "cos_theta_day", cos_theta_day, True)
    return X1


# Polar to cartesian wind coordinates
def wd_to_coord(X1):
    
    for index, row in X1.iterrows():
        if row['wd'] == 'E': 
            X1.at[index,'wdc'] = 2*np.pi*(0/16)
        elif row['wd'] == 'ENE': 
            X1.at[index,'wdc'] = 2*np.pi*(1/16)
        elif row['wd'] == 'NE': 
            X1.at[index,'wdc'] = 2*np.pi*(2/16)
        elif row['wd'] == 'NNE': 
            X1.at[index,'wdc'] = 2*np.pi*(3/16)
        elif row['wd'] == 'N': 
            X1.at[index,'wdc'] = 2*np.pi*(4/16)
        elif row['wd'] == 'NNW': 
            X1.at[index,'wdc'] = 2*np.pi*(5/16)
        elif row['wd'] == 'NW': 
            X1.at[index,'wdc'] = 2*np.pi*(6/16)
        elif row['wd'] == 'WNW': 
            X1.at[index,'wdc'] = 2*np.pi*(7/16)
        elif row['wd'] == 'W': 
            X1.at[index,'wdc'] = 2*np.pi*(8/16)
        elif row['wd'] == 'WSW': 
            X1.at[index,'wdc'] = 2*np.pi*(9/16)
        elif row['wd'] == 'SW': 
            X1.at[index,'wdc'] = 2*np.pi*(10/16)
        elif row['wd'] == 'SSW': 
            X1.at[index,'wdc'] = 2*np.pi*(11/16)
        elif row['wd'] == 'S': 
            X1.at[index,'wdc'] = 2*np.pi*(12/16)
        elif row['wd'] == 'SSE': 
            X1.at[index,'wdc'] = 2*np.pi*(13/16)
        elif row['wd'] == 'SE': 
            X1.at[index,'wdc'] = 2*np.pi*(14/16)
        else : 
            X1.at[index,'wdc'] = 2*np.pi*(15/16)
    
    wd = X1.loc[:,['wdc']].to_numpy()
    wd_speed = X1.loc[:,['WSPM']].to_numpy()
    sin_wd = np.sin(wd) * wd_speed
    cos_wd = np.cos(wd) * wd_speed
    X1.insert(X1.shape[1], "sin_wd", sin_wd, True)
    X1.insert(X1.shape[1], "cos_wd", cos_wd, True)
    return X1


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


# Correlation
def project_correlation(X, Y, fig=True):
    print('Correlation...')
    
    XY = pd.concat([X, Y], axis=1)
    corr = XY.corr()
    
    plt.figure(figsize=(25,15))
    sns.heatmap(corr, annot=True, cmap=plt.cm.Reds).set_title('Correlation')
    plt.show()
    
    return corr


# Mutual information (buggy, not the same)
def project_mutual_info(X, Y):
    print('Mutual information...')
    
    XY = pd.concat([X, Y], axis=1)
    XY_col = XY.columns
    XY = XY.values
    n = XY.shape[1]
    mut = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            mut[i,j] = mutual_info_regression(XY[:,i].reshape((XY.shape[0],1)), XY[:,j], random_state=0) if j >= i else mut[j,i]
    
    mut = pd.DataFrame(mut)
    mut.columns = XY_col
    mut.index = mut.columns
    
    plt.figure(figsize=(25,15))
    sns.heatmap(mut, annot=True, cmap=plt.cm.Reds).set_title('Mutual Information')
    plt.show()
    
    return mut


#Linear regression
def project_linear_regression(X, Y):
    reg = LinearRegression().fit(X, Y)
    score = reg.score(X, Y)
    params = reg.get_params()
    
    Y_pred = reg.predict(X)
    
    return rmse(Y, Y_pred)

def project_KNN(X_train, y_train, X_test, y_test):
    n = 50 
    neighbors = np.arange(1,n+1,1)
    mse_knn = np.zeros(n)
    for i in range(0,n):
        knn = KNeighborsRegressor(n_neighbors=neighbors[i], metric='euclidean')
        y_knn = knn.fit(X_train, y_train).predict(X_test)
        mse_knn[i] = np.square(np.subtract(y_knn, y_test)).mean()
    
    fig = plt.figure(figsize=(6,4))
    
    plt.plot(neighbors,mse_knn, 'b')
    plt.xlabel('neighbours')
    plt.ylabel('mse_knn')
    
    plt.show()
    
    #plot_file = "./knn_mse.eps"
    #fig.savefig(plot_file, facecolor='w', edgecolor='w', format='eps', bbox_inches='tight', pad_inches=0)
    
    n = 50 
    alpha = np.logspace(-4.0, 4.0, num=n)
    mse_lasso = np.zeros(n)
    for i in range(0,n):
        clf = linear_model.Lasso(alpha=alpha[i])
        y_lasso = clf.fit(X_train, y_train).predict(X_test)
        mse_lasso[i] = np.square(np.subtract(y_lasso, y_test)).mean()
    
    fig = plt.figure(figsize=(6,4))
    
    plt.plot(alpha,mse_lasso, 'b')
    plt.xlabel('alpha')
    plt.ylabel('mse_lasso')
    plt.xscale('log')
    plt.show()
    
    pca = PCA(n_components=4)
    X1_PCA = pca.fit_transform(X1)
    print(pca.explained_variance_ratio_)
    
    X_PCA_train, X_PCA_test, y_PCA_train, y_PCA_test = train_test_split(X1_PCA, Y1, random_state=1)
    
    n = 50 
    neighbors = np.arange(1,n+1,1)
    mse_PCA_knn = np.zeros(n)
    for i in range(0,n):
        knn = KNeighborsRegressor(n_neighbors=neighbors[i], metric='euclidean')
        y_PCA_knn = knn.fit(X_PCA_train, y_train).predict(X_PCA_test)
        mse_PCA_knn[i] = np.square(np.subtract(y_PCA_knn, y_PCA_test)).mean()
    
    fig = plt.figure(figsize=(6,4))
    
    plt.plot(neighbors,mse_PCA_knn, 'b')
    plt.xlabel('neighbours')
    plt.ylabel('mse_PCA_knn')
    
    plt.show()




# Data building
if 'Xpd' not in locals():
    print('Loading...')
    Xpd, X_full, Y = load_data()

#print(np.max(X_full,1))
#print(np.min(X_full,1))
nd, nf = X_full.shape # nd: nbr of data, nf: nbr of features

### Features selection

corr = project_correlation(X_full, Y)

mut = project_mutual_info(X_full, Y)


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

X = X_full.values[:, mut.values[:-1,-1] > 0.1]

# Data separation between training and validation sets
frac_valid = 0.1
X_train, X_val, Y_train, Y_val = train_test_split(X, Y.values, random_state=1)


RMSE_lin_reg = project_linear_regression(X_train, Y_train)

project_KNN(X_train, X_val, Y_train, Y_val)