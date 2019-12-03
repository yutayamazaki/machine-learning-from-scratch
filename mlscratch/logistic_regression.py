import numpy as np
from sklearn.exceptions import NotFittedError

from mlscratch import losses
from mlscratch.activations import sigmoid


class LogisticRegression(object):
    """ A simple LogisticRegression implementation with NumPy.

    Args:
        lr (float): Learning rate.
        num_iterations (int): A number of iterations.
        threshold (float): A threshold for classify.
        verbose (bool): A verbosity parameter.
    """

    def __init__(
        self,
        lr=0.001,
        num_iterations=10000,
        threshold=0.5,
        verbose=True,
    ):
        self.lr = lr
        self.num_iterations = num_iterations
        self.threshold = threshold
        # Parameters to train.
        self.theta = None
        self.verbose = verbose

    @staticmethod
    def _initialize_parameters(X):
        num_features = X.shape[1]
        limit = 1 / np.sqrt(num_features)
        params = np.random.uniform(-limit, limit, (num_features, ))
        return params

    def fit(self, X, y):
        assert X.ndim == 2, f'X.ndim must be 2. But got {X.ndim}.'
        assert y.ndim == 1, f'y.ndim must be 1. But got {y.ndim}.'
        self.theta = self._initialize_parameters(X)
        data_size = len(y)

        for n_iter in np.arange(self.num_iterations):
            # Make a new prediction.
            y_pred = sigmoid(X.dot(self.theta))
            # Update parameters.
            self.theta = \
                self.theta - self.lr*(1/data_size)*(X.T.dot(y_pred - y))
            # Calculate BinaryCrossEntropy.
            # bce = losses.binary_cross_entropy(y, y_pred)
            bce = losses.binary_cross_entropy(y, y_pred)
            if self.verbose:
                print(f'num_iterations: {n_iter}\t BCE: {bce}')

        return self

    def predict(self, X):
        assert X.ndim == 2, f'X.ndim must be 2. But got {X.ndim}.'
        if self.theta is None:
            raise NotFittedError(f'{self.__class__.__name__} is not fitted.'
                                 f' You must call fit before call predict.')
        y_pred = sigmoid(X.dot(self.theta)) >= self.threshold
        return (y_pred.astype('int'))

    def predict_proba(self, X):
        assert X.ndim == 2, f'X.ndim must be 2. But got {X.ndim}.'
        if self.theta is None:
            raise NotFittedError(f'{self.__class__.__name__} is not fitted.'
                                 f' You must call fit before call predict.')
        return sigmoid(X.dot(self.theta))
