import unittest

import numpy as np
from sklearn.datasets import load_boston

from mlscratch.models.linear import Ridge


class TestRidge(unittest.TestCase):

    def _load_data(self):
        X, y = load_boston(return_X_y=True)
        X, y = X[:10], y[:10]
        return X, y

    def test_init(self):
        model = Ridge()
        self.assertEqual(getattr(model, 'alpha'), 1.0)

    def test_fit(self):
        X, y = self._load_data()
        model = Ridge()
        ret = model.fit(X, y)
        self.assertIsInstance(ret, Ridge)
        self.assertEqual(model.coef_.shape, (X.shape[1], ))
        self.assertEqual(model.intercept_.shape, ())

    def test_predict(self):
        X, y = self._load_data()
        model = Ridge()
        model.fit(X, y)
        y_pred = model.predict(X)

        self.assertIsInstance(y_pred, np.ndarray)
        self.assertEqual(y_pred.shape, (X.shape[0], ))
