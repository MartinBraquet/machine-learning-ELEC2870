# -*-coding:Utf-8 -*

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def show_quantization(X, X_c, use_seaborn=False):
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
    
    if use_seaborn:
        # Moving the data into a DataFrame for seaborn compatibility
        # (Seaborn allows for more eye candy when making graphs)   
        X = pd.DataFrame(data=X, columns=["X1", "X2"])
        X["centroid"] = closest
        X_c = pd.DataFrame(data=X_c, columns=["X1", "X2"])
        X_c["centroid"] = range(n_centroids)
        sns.set() # Set seaborn defaults before call to figure
    else:
        sns.set_style("white") # Restore to 'default' mpl colors
        
    # Showing the vector quantization
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    if use_seaborn:
        palette = sns.color_palette("deep", n_centroids)
        sns.scatterplot(data=X, x="X1", y="X2", hue="centroid", palette=palette)
        sns.scatterplot(data=X_c, x="X1", y="X2", hue="centroid", palette=palette,
                        s=150, marker='d', edgecolor='black', alpha=0.8)
        ax.legend_.remove()
    else:
        plt.scatter(X[:,0], X[:,1], c=closest, alpha=0.6)
        plt.scatter(X_c[:,0], X_c[:,1], c=range(n_centroids), marker='d', edgecolor='black', s=150, alpha=0.8)
        plt.grid()
    
    
 
    ax.set_title(f"Vector quantization ({n_points} samples - {n_centroids} centroids)", fontsize=15)
    ax.set_xlabel("$X_1$", fontsize=15)
    ax.set_ylabel("$X_2$", fontsize=15)

    plt.show()
    plt.close()

