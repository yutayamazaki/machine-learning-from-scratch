import unittest

import numpy as np

from mlscratch.models.perceptron import Perceptron



class TestPerceptron(unittest.TestCase):

    def setUp(self):
        self.model = Perceptron()

    def test_init_weights(self):
        """ Check return value and its dtype. """
        result = self.model._init_weights([[0, 1], [0, 1]])
        expected = np.array([0, 0])

        self.assertAlmostEqual(result.all(), expected.all())
        self.assertIsInstance(result, np.ndarray)

    def test_forward(self):
        """ Check return value and its dtype. """
        x = np.array([[0, 1], [0, 1]])
        weight = self.model._init_weights(x)
        result = self.model.forward(x[0], weight)
        expected = np.float(0.)
        self.assertAlmostEqual(result, expected)

    def test_forward_raise_if_ndim_eq_one(self):
        """ Check it raises ValueError if x.ndim is not 1. """
        # Make data of X.ndim == 2.
        x = np.array([[0, 1], [0, 1]])
        weight = self.model._init_weights(x)
        with self.assertRaises(ValueError):
            self.model.forward(x, weight)

    def test_predict(self):
        """ Check return values and its dtype. """
        X_train = np.array([
            [0, 0, 1],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ])
        y_train = np.array([0, 0, 0, 1])
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_train)

        self.assertIsInstance(y_pred, np.ndarray)
        self.assertAlmostEqual(y_pred.all(), y_train.all())
