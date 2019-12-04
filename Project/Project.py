'''
Questions: 
- Validation set for feature selection?
- How to characterize wd and time?
- Normalization, how it works?
- MI, how it works?
'''

import numpy as np
from sklearn.feature_selection import mutual_info_regression
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from datetime import datetime

def load_data(alpha=0.1):
    Xpd = pd.read_csv("Datasets/X1.csv",sep = ',')
    Xpd = time_to_coord(Xpd, alpha)
    Xpd = wd_to_coord(Xpd)

    X_full = Xpd
    X_full = X_full.drop(columns="wd")
    X_full = X_full.drop(columns="wdc")
    X_full = X_full.drop(columns="year")
    X_full = X_full.drop(columns="month")
    X_full = X_full.drop(columns="day")
    X_full = X_full.drop(columns="hour")
    X_full = X_full.to_numpy()
    
    Y = pd.read_csv("Datasets/Y1.csv", header=None).values[:,0]
    
    return Xpd, X_full, Y

def time_to_coord(X1, alpha):
    sec_per_year = (datetime(2014,1,1)-datetime(2013,1,1)).total_seconds()
    z = np.zeros(X1.shape[0])
    theta = np.zeros(X1.shape[0])
    for index, row in X1.iterrows():
        t = datetime(row['year'], row['month'], row['day'], row['hour'])
        seconds_tot = (t-datetime(2013,3,1)).total_seconds()
        theta[index] = 2*np.pi*seconds_tot/sec_per_year
        z[index] = alpha * seconds_tot/sec_per_year
    
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    X1.insert(X1.shape[1], "z", z, True) 
    X1.insert(X1.shape[1], "sin_theta", sin_theta, True) 
    X1.insert(X1.shape[1], "cos_theta", cos_theta, True) 
    return X1


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
    wd_1 = np.sin(wd)
    wd_2 = np.cos(wd)
    X1.insert(X1.shape[1], "wd_1", wd_1, True)
    X1.insert(X1.shape[1], "wd_2", wd_2, True)
    return X1

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

# Data building
if 'Xpd' in locals():
    print('Loading...')
    Xpd, X_full, Y = load_data()
    X_full = preprocessing.scale(X_full) # Smaller error with scale instead of normalize

#print(np.max(X_full,1))
#print(np.min(X_full,1))
nd, nf = X_full.shape # nd: nbr of data, nf: nbr of features

#np.random.shuffle(X_full)

### Features selection

# Correlation
corr = np.zeros((nf,))
for i in range(nf):
    corr[i] = np.corrcoef(X_full[:,i], Y)[0,1]
print('Correlation:', corr)

# Mutual information (buggy, not the same)
mut = mutual_info_regression(X_full, Y)
print('Mutal information:', mut)

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

X = X_full[:, mut > 0.01]

# Data separation between training and validation sets
frac_valid = 0.1
X_train = X[int(frac_valid * nd):,:]
Y_train = Y[int(frac_valid * nd):]
X_val = X[:int(frac_valid * nd),:]
Y_val = Y[:int(frac_valid * nd)]

#Linear regression
reg = LinearRegression().fit(X_train, Y_train)
score = reg.score(X_train, Y_train)
params = reg.get_params()

Y_pred = reg.predict(X_train)

RMSE_lin_reg = rmse(Y_pred, Y_train)
