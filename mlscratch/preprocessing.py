import numpy as np


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
