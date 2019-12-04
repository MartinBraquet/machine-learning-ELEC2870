# Imports

import scipy.io
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib as mpl


### Initialization
def get_inits(X, Q, method = "sample"):
    inits = {}
    mini = np.min(X, axis=0)
    maxi = np.max(X, axis=0)
    
    n_feats = X.shape[-1]
    rand_array = []
    for i in range(n_feats):
        rand_array.append(
            np.random.uniform(mini[i],maxi[i],Q)
        )
    inits["random"] = np.array(rand_array).T
    inits["sample"] = np.array(random.sample(X.tolist(), Q))

    return inits[method]

### Learning
def comp_learning(X, Y, n_epochs=100, alpha=0.1, beta=0.99, min_epsilon=1e-3):
    X = X.copy()
    for epoch in range(n_epochs):
        prev_Y = Y.copy()
        
        # Shuffle Data
        np.random.shuffle(X)
        
        # Use up every data point
        for xp in X:
            
            # Find the closest centroid
            index = np.argmin(np.sum((xp-Y)**2, axis=-1))
            
            # Update closest centroid
            yq = Y[index,:]
            Y[index,:] = yq + alpha * (xp - yq)
                
        # Update LR
        alpha = alpha*beta / (alpha+beta)
        
        # 'Intelligent' stopping criterion
        if np.mean(np.abs(prev_Y-Y)) < min_epsilon:
            print('Stopped at epoch ' + str(epoch))
            break

    return Y

### Visualization
def show_quantization(X, X_c):
    """
    Visualise a vector quantization.

       show_quantization(X, X_c)

    Inputs:
      - X: numpy.ndarray containing the instances (p x 2)
      - X_c: numpy.ndarray containing the centroids (q x 2)

    A figure is created and the p instances are shown as dots,
    whereas the q centroids are shown as diamonds.
    The instances' colors show their appartenance to the centroid 
    of the same color

    Authors: Cyril de Bodt (2016) - cyril.debodt@uclouvain.be
             Antoine Vanderschueren (2019) - antoine.vanderschueren@uclouvain.be
    Version: 08-10-2019

    """

    # Checking the arguments
    if not (isinstance(X, np.ndarray) and isinstance(X_c, np.ndarray)):
        raise ValueError("""X and X_c must be numpy.ndarray""")
    if not ((len(X.shape) == 2) and (len(X_c.shape) == 2)):
        raise ValueError("""X and X_c must be numpy.ndarray with 2 dimensions""")
    if not ((X.shape[1] == 2) and (X_c.shape[1] == 2)):
        raise ValueError("""X and X_c must be numpy.ndarray with 2 dimensions and 2 columns""")

    # Finding the index of the nearest centroid in X_c for each point in X
    n_centroids = len(X_c)
    n_points = len(X)
    distances = np.sum((np.repeat(X, n_centroids, axis=0).reshape(n_points, n_centroids, 2) - X_c)**2, axis=-1)
    closest = np.argmin(distances, axis=-1)
        
    # Showing the vector quantization
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.scatter(X[:,0], X[:,1], c=closest, alpha=0.6)
    plt.scatter(X_c[:,0], X_c[:,1], c=range(n_centroids), marker='d', edgecolor='black', s=150, alpha=0.8)
    plt.grid()
    
    ax.set_title("Vector quantization ("+str(n_points)+" samples - "+str(n_centroids)+" centroids)", fontsize=15)
    ax.set_xlabel("$X_1$", fontsize=15)
    ax.set_ylabel("$X_2$", fontsize=15)

    plt.show()
    plt.close()