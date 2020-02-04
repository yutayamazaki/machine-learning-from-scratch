import sys
sys.path.append('../')

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris

from mlscratch.decomposition.pca import PCA

if __name__ == '__main__':
    X, y = load_iris(return_X_y=True)

    X_reduced = PCA(num_components=2).fit_transform(X)

    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y)
    plt.colorbar()
    plt.title('Scatter plot with iris dataset')
    plt.show()
