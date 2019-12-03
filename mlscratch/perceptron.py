import numpy as np

from mlscratch.activations import relu


class Perceptron:

    def __init__(self, eta: float = 0.1, num_epochs: int = 100):
        self.eta = eta
        self.num_epochs = num_epochs

    @staticmethod
    def _init_weights(X):
        X = np.array(X)
        num_features = X.shape[1]
        return np.zeros(num_features)

    @staticmethod
    def forward(x, weight):
        if x.ndim != 1:
            raise ValueError(f'x.ndim must be 1, but got {x.ndim}')
        return relu(np.dot(x, weight))

    def fit(self, X, y):
        if X.ndim != 2:
            raise ValueError(f'X.ndim must be 2, but got {X.ndim}')
        self.weight = self._init_weights(X)
        for i in range(self.num_epochs):
            for x_iter, y_iter in zip(X, y):
                y_pred = self.forward(x_iter, self.weight)
                self.weight += (y_iter - y_pred) * x_iter * self.eta
        return self

    def predict(self, X):
        y_pred = []
        for x in X:
            y_pred.append(self.forward(x, self.weight))
        return np.array(y_pred)
