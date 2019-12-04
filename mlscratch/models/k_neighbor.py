import numpy as np


class KNN:

    def __init__(self, k: int = 5):
        self.k = k
        self.X = None
        self.y = None

    @staticmethod
    def euclidean_distance(m, n):
        return np.sqrt(np.power(m - n, 2)).mean()

    def fit(self, X, y):
        if X.ndim != 2:
            raise ValueError(f'X.ndim must be 2, but got {X.ndim}')

        self.X = X
        self.y = y
        return self

    def predict(self, X):
        if (self.X is None) or (self.y is None):
            raise ValueError(f'{self} is not fitted.')

        y_pred = []
        for x in X:
            distances = []
            for x_train in self.X:
                distance = self.euclidean_distance(x, x_train)
                distances.append(distance)
            # Get indices of nearest neighbors.
            indices = np.argsort(distances)[:self.k]
            # Make prediction!!
            y_pred_iter = np.mean(self.y[indices])
            y_pred.append(y_pred_iter)

        return np.array(y_pred)
