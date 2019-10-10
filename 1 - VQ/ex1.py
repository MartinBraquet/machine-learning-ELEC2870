# -*-coding:Utf-8 -*

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io 
from show_quantization import *

#plt.clf()

data1 = scipy.io.loadmat('data/dataset_1.mat')
X1 = data1['X']

'''
data2 = scipy.io.loadmat('data/dataset_2.mat')
X2 = data2['X']

data3 = scipy.io.loadmat('data/dataset_3.mat')
X3 = data3['X']

data4 = scipy.io.loadmat('data/dataset_4.mat')
X4 = data4['X']


fig, axs = plt.subplots(nrows=2,ncols=2)
fig.suptitle('Just some random samples')

axs[0,0].scatter(X1[:,0],X1[:,1])
axs[0,0].set_title('Sample 1')
axs[0,0].set_ylabel('y1')
axs[0,0].set_xlabel('x1')

axs[0,1].scatter(X2[:,0],X2[:,1])
axs[0,1].set_title('Sample 2')
axs[0,1].set_ylabel('y2')
axs[0,1].set_xlabel('x2')

axs[1,0].scatter(X3[:,0],X3[:,1])
axs[1,0].set_title('Sample 3')
axs[1,0].set_ylabel('y3')
axs[1,0].set_xlabel('x3')

axs[1,1].scatter(X4[:,0],X4[:,1])
axs[1,1].set_title('Sample 4')
axs[1,1].set_ylabel('y4')
axs[1,1].set_xlabel('x4')
'''

#plt.rcParams["figure.figsize"] = (40,40) # remove to see overlapping subplots

#plt.show()

#plt.scatter(X1[:,0],X1[:,1])
#plt.rcParams["figure.figsize"] = (10,10)
#plt.show()

def competitive_learning(x, Q, init):
    
    a = 0.8
    b = 10
    
    P = np.size(x,0)
    D = np.size(x,1)
    
    maxs = np.max(x,0)
    mins = np.min(x,0)
    print(maxs)
    print(mins)
    
    cent = np.zeros((Q,D))
    
    if init == 1:
        for i in range(D):
            cent[:,i] = np.random.uniform(low=mins[i], high=maxs[i], size=(Q,))
    else:
        pass
    
    print(cent)
    
    error = 0
    
    while (True):
        a = a * b / (a + b)
        
        np.random.shuffle(x)
        
        for i in range(P):
            (index, dist) = find_closest(cent, x[i,:], Q, D)
            # show_quantization(np.array([x[i,:],[0,0]]), cent)
            # print(cent[index,:])
            # print(x[i,:])
            cent[index,:] += a * (x[i,:] - cent[index,:])
            
        old_error = error
        error = 0
        for i in range(P):
            (index, dist) = find_closest(cent, x[i,:], Q, D)
            error += dist
        error = error / P
            
        print(error)
        print(old_error)
        print(abs(error - old_error) / error)
        if (abs(error - old_error) / error < 0.000001):
            break
            
    return cent
            
        
def find_closest(v, x, Q, D):
    dist = np.zeros((Q,))
    for j in range(D):
        # print(v[:,j] - x[j])
        dist += (v[:,j] - x[j])**2
    index = np.argmin(dist)
    return (index, dist[index])
        
centroids = competitive_learning(X1, 4, 1)

show_quantization(X1, centroids)