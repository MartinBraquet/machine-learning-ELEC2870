# -*-coding:Utf-8 -*

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri
from scipy.spatial import Delaunay

def visualize(X, T):
    """
    Visualize the values of a function with bidimensional inputs.

      visualize(X, T)

    Inputs:
      - X: numpy.ndarray containing the instances (n x 2)
      - T: numpy.ndarray containing the function values (n x 1)

    Author: Cyril de Bodt (2016) - cyril.debodt@uclouvain.be
    Version: 10-11-2016-14:00

    """

    # Checking the arguments
    if not (isinstance(X, np.ndarray) and isinstance(T, np.ndarray)):
        raise ValueError("""X and T must be numpy.ndarray""")
    if not (len(X.shape) == 2):
        raise ValueError("""X must have 2 dimensions, and T only 1.""")
    if not ((X.shape[1] == 2) and (T.shape[0] == X.shape[0])):
        raise ValueError("""X must have 2 columns and the same number of lines as T""")

    T = T.flatten()

    # Triangulate parameter space to determine the triangles
    tri = Delaunay(X)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    # The triangles in parameter space determine which x, y, z points are
    # connected by an edge
    #ax.plot_trisurf(x, y, z, triangles=tri.triangles, cmap=plt.cm.Spectral)
    ax.plot_trisurf(X[:,0], X[:,1], T, triangles=tri.simplices, cmap=plt.cm.Spectral)

    ax.set_xlabel("$X_1$", fontsize=15)
    ax.set_ylabel("$X_2$", fontsize=15)
    ax.set_zlabel("$T$", fontsize=15)

    plt.show()
    plt.close()