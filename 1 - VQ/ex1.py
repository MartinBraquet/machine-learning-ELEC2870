# -*-coding:Utf-8 -*

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io 
from show_quantization import show_quantization

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
    
    X_c = np.zeros((Q,D))
    
    if init == 1:
        for i in range(D):
            X_c[:,i] = np.random.uniform(low=mins[i], high=maxs[i], size=(Q,))
    else:
        pass
    
    print(X_c)
    
    error = 0
    
    while (True):
        a = a * b / (a + b)
        
        np.random.shuffle(x)
        
        for i in range(P):
            distances = np.sum((X_c - x[i,:])**2, 1)
            index = np.argmin(distances)
            # show_quantization(np.array([x[i,:],[0,0]]), X_c)
            # print(X_c[index,:])
            # print(x[i,:])
            X_c[index,:] += a * (x[i,:] - X_c[index,:])
            
        old_error = error
            
        distances = np.sum((np.repeat(x, Q, axis=0).reshape(P, Q, 2) - X_c)**2, axis=-1)
        error = np.sum(np.min(distances, axis=-1)) / P
        
        #print(error)
        #print(old_error)
        print(abs(error - old_error) / error)
        if (abs(error - old_error) / error < 0.001):
            break
            
    return X_c
            
'''
def find_closest(X_c, x, Q, D):
    dist = np.zeros((Q,))
    for j in range(D):
        # print(v[:,j] - x[j])
        dist += (X_c[:,j] - x[j])**2
    distances = np.sum((np.repeat(X, n_X_croids, axis=0).reshape(n_points, n_X_croids, 2) - X_c)**2, axis=-1)
    closest = np.argmin(distances, axis=-1)
    index = np.argmin(dist)
    return (index, dist[index])
'''

X_c = competitive_learning(X1, 4, 1)

show_quantization(X1, X_c, use_seaborn=True)

print('test')

show_quantization(X1, X_c)

plt.show()
