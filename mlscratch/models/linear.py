import numpy as np

from mlscratch.models.losses import mean_squared_error


class LinearRegression(object):
    """ A simple implementation of LinearRaegresion.

    Args:
        num_iterations (int): A number of iterations to train.
        lr (float): A learning rate to update parameters in each iteration.
    """

    def __init__(self, num_iterations=1000, lr=0.01):
        self.num_iterations = num_iterations
        self.lr = lr
        self.w = None
        self.train_metric_list = []

    @staticmethod
    def _initialize_parameters(X):
        """Initialize parameters with bias.

        Args:
            X (np.ndarray): With dim 2.

        Returns:
            np.ndarray: Initialized parameters.
        """
        num_features = X.shape[1]
        num_parameters = num_features + 1
        limit = 1 / np.sqrt(num_parameters)
        params = np.random.uniform(-limit, limit, (num_parameters, ))
        return params

    @staticmethod
    def _expand_column(X):
        """Expand matrix with ones to use bias.

        Args:
            X (np.ndarray): With shape (n_rows, n_cols).

        Return:
            np.ndarray: With shape (n_rows, n_cols+1)
        """
        data_size = X.shape[0]
        one_array = np.ones(data_size).reshape(data_size, 1)
        X_expand = np.concatenate([one_array, X], axis=1)
        return X_expand

    def fit(self, X, y):
        """Fit a model to given dataset."""
        self.w = self._initialize_parameters(X)
        data_size = X.shape[0]

        X_expand = self._expand_column(X)
        for idx in range(self.num_iterations):
            y_pred = self.predict(X)
            dw = np.dot((y - y_pred), X_expand) / data_size
            self.w += self.lr * dw
            loss = mean_squared_error(y, y_pred)
            self.train_metric_list.append(loss)
        return self

    def predict(self, X):
        y_pred = self.w[0] + np.dot(X, self.w[1:])
        return y_pred


class Ridge:

    """ Ridge regression.

    Args:
        alpha (float): A weight of regularization.

    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X, y):
        # 1-padding for first column
        X = np.insert(X, 0, 1, axis=1)
        identity_mat = np.eye(X.shape[1])
        W = np.linalg.inv(
            X.T.dot(X) + self.alpha * identity_mat
        ).dot(X.T).dot(y)
        self.coef_ = W[1:]
        self.intercept_ = W[0]
        return self

    def predict(self, X):
        return X.dot(self.coef_) + self.intercept_
