import numpy as np


def minmaxscaler(matrix):
    x_min = np.min(matrix, axis=0)
    x_max = np.max(matrix - x_min, axis=0)

    scaled = (matrix - x_min) / x_max
    return scaled


class MinMaxScaler:

    def fit(self, X, y=None):
        self.x_min_ = np.min(X, axis=0)
        self.x_max_ = np.max(X - self.x_min_, axis=0)
        return self

    def transform(self, X, y=None):
        return (X - self.x_min_) / self.x_max_

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)


if __name__ == '__main__':

    mat = np.array([
        [0, 1, 2],
        [5, 0, 1]
    ])
    mat = minmaxscaler(mat)
    print(mat)

    mat = np.array([
        [0, 1, 2],
        [5, 0, 1]
    ])
    print(MinMaxScaler_().fit_transform(mat))
