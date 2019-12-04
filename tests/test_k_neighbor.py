import unittest

import numpy as np

from mlscratch.k_neighbor import KNN


class TestKNN(unittest.TestCase):

    def setUp(self):
        self.k = 5
        self.model = KNN(k=self.k)

    def test_constructor(self):
        """ Check attributes. """
        self.assertEqual(self.model.k, self.k)
        self.assertIsNone(self.model.X)
        self.assertIsNone(self.model.y)

    def test_euclidean_distance(self):
        """ Check calculated correctly. """
        x1 = np.array([0, 1])
        x2 = np.array([0, 1])
        distance = self.model.euclidean_distance(x1, x2)
        self.assertAlmostEqual(distance, 0.0)
        self.assertIsInstance(distance, np.float64)

        x3 = np.array([0, 1])
        x4 = np.array([0, 0])
        distance = self.model.euclidean_distance(x3, x4)
        self.assertAlmostEqual(distance, 0.5)
        self.assertIsInstance(distance, np.float64)

    def test_fit_return_value(self):
        """ Check dtype of return value. """
        X = np.array([
            [0, 1],
            [2, 2]
        ])
        y = np.array([0, 1])
        result = self.model.fit(X, y)
        self.assertIsInstance(result, KNN)

    def test_fit_invalid_dim_raise_value_error(self):
        """ If X.ndim!=2, KNN raises ValueError. """
        X_invalid = np.array([0, 1])
        y = np.array([0, 1])
        with self.assertRaises(ValueError):
            self.model.fit(X_invalid, y)

    def test_predict(self):
        """ Check prediction result. """
        X = np.array([
            [0, 1],
            [2, 2]
        ])
        y = np.array([0, 1])
        self.model.fit(X, y)
        y_pred = self.model.predict(X)

        self.assertEqual(y_pred.shape, y.shape)
        self.assertIsInstance(y_pred, np.ndarray)
