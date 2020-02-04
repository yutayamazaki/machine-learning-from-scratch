import unittest

import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler

from mlscratch.models.linear import LinearRegression


class LinearRegressionTests(unittest.TestCase):

    X, y = load_boston(return_X_y=True)
    X = StandardScaler().fit_transform(X)

    def test_initialize_parameters(self):
        """ Check the shape and dtype of returned value. """
        model = LinearRegression()
        w = model._initialize_parameters(self.X)
        expected_shape = (self.X.shape[1] + 1, )
        self.assertEqual(w.shape, expected_shape)
        self.assertIsInstance(w, np.ndarray)

    def test_predict(self):
        """ Check the shape and dtype of returned value. """
        model = LinearRegression()
        model.w = model._initialize_parameters(self.X)
        y_pred = model.predict(self.X)
        expected_shape = (self.X.shape[0], )
        self.assertEqual(y_pred.shape, expected_shape)
        self.assertIsInstance(y_pred, np.ndarray)

    def test_fit(self):
        """ Check returned value. """
        model = LinearRegression()
        ret = model.fit(self.X, self.y)
        self.assertIsInstance(ret, LinearRegression)
        self.assertIsInstance(model.train_metric_list, list)
        self.assertIsInstance(model.w, np.ndarray)
