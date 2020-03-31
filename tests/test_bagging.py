import unittest

import numpy as np
from sklearn.datasets import load_breast_cancer

from mlscratch.models.bagging import Bagging, bootstrap_sampling


class BootstrapSamplingTests(unittest.TestCase):

    def test_return(self):
        """ Check return value """
        X, y = load_breast_cancer(return_X_y=True)
        X_sample, y_sample = bootstrap_sampling(X, y)
        assert X_sample.shape == X.shape
        assert y_sample.shape == y.shape
        assert isinstance(X_sample, np.ndarray)
        assert isinstance(y_sample, np.ndarray)


class BaggingTests(unittest.TestCase):

    def setUp(self):
        self.X, self.y = load_breast_cancer(return_X_y=True)

    def test_fit(self):
        num_estimators: int = 2
        model = Bagging(num_estimators=num_estimators)
        ret = model.fit(self.X[2:], self.y[2:])
        assert ret == model
        assert len(model.estimators) == num_estimators

    def test_predict(self):
        num_estimators: int = 2
        model = Bagging(num_estimators=num_estimators)
        model.fit(self.X[:2], self.y[:2])
        y_pred = model.predict(self.X[:2])
        assert y_pred.shape == (2, )
        assert isinstance(y_pred, np.ndarray)
