import unittest

import numpy as np
from sklearn.datasets import load_breast_cancer

from mlscratch.models.logistic_regression import LogisticRegression



class TestLogisticRegression(unittest.TestCase):

    def setUp(self):
        self.X, self.y = load_breast_cancer(return_X_y=True)

        model = LogisticRegression(num_iterations=10, verbose=False)
        self.fitted = model.fit(self.X, self.y)

    def test_initialize_parameters(self):
        model = LogisticRegression()
        params = model._initialize_parameters(self.X)

        # Check the shape of returned array.
        expected = (self.X.shape[1], )
        self.assertEqual(params.shape, expected)

    def test_fit(self):
        model = LogisticRegression(num_iterations=10, verbose=False)
        # theta must be None before fit.
        self.assertIsNone(model.theta)
        ret = model.fit(self.X, self.y)
        # Return LogisticRegression itself.
        self.assertIsInstance(ret, LogisticRegression)
        # theta must be not None after fit.
        self.assertIsNotNone(model.theta)

    def test_predict(self):
        y_pred = self.fitted.predict(self.X)
        # Check return value.
        self.assertIsInstance(y_pred, np.ndarray)
        self.assertEqual(y_pred.shape, self.y.shape)

    def test_predict_proba(self):
        y_proba = self.fitted.predict_proba(self.X)
        self.assertIsInstance(y_proba, np.ndarray)
        self.assertEqual(y_proba.shape, self.y.shape)
