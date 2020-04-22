import unittest

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder

from mlscratch.models.multilayer_perceptron import MLP



class TestMLP(unittest.TestCase):

    def _load_data(self):
        X, y = load_iris(return_X_y=True)
        X, y = X[:10], y[:10]
        y = OneHotEncoder().fit_transform(y.reshape(-1, 1))
        y = y.toarray()
        return X, y

    def test_fit(self):
        model = MLP(num_hidden=8)
        X, y = self._load_data()
        ret = model.fit(X, y)
        self.assertIsInstance(ret, MLP)

    def test_init_weights(self):
        num_hidden = 4
        model = MLP(num_hidden=num_hidden)
        X, y = self._load_data()
        model._init_weights(X, y)

        # first layer
        self.assertIsInstance(model.w1, np.ndarray)
        self.assertSequenceEqual(model.w1.shape, (X.shape[1], num_hidden))
        self.assertIsInstance(model.b1, np.ndarray)
        self.assertSequenceEqual(model.b1.shape, (1, num_hidden))
        # second layer
        self.assertIsInstance(model.w2, np.ndarray)
        self.assertSequenceEqual(model.w2.shape, (num_hidden, 1))
        self.assertIsInstance(model.b2, np.ndarray)
        self.assertSequenceEqual(model.b2.shape, (1, 1))

    def test_predict(self):
        num_hidden = 4
        model = MLP(num_hidden=num_hidden)
        X, y = self._load_data()
        model.fit(X, y)
        y_pred = model.predict(X)
        self.assertIsInstance(y_pred, np.ndarray)
        self.assertSequenceEqual(y_pred.shape, (len(X), 1))
