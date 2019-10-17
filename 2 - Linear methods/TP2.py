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
title = ["age", "sex", "bmi", "blood_pressure", "serum_1", "serum_2", "serum_3", "serum_4", "serum_5", "serum_6"]

fig, axs = plt.subplots(nrows=4,ncols=3)
#fig.suptitle('Just some random samples')

for i in range (10):
    d = {title[i]: X[:,i], 't': t[:,0]}
    Xpd = pd.DataFrame(data=d, columns=[title[i], "t"])
    sns.set() # Set seaborn defaults before call to figure
    
    # Showing the vector quantization
    sns.scatterplot(data=Xpd, x = title[i], y = "t", ax=axs[(i-i%3)//3][i%3])
    

plt.show()
plt.close()