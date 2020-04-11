import unittest

import numpy as np
from sklearn.datasets import load_iris

from mlscratch.models import decision_tree


class DecisionTreeClassifierTests(unittest.TestCase):

    def setUp(self):
        self.model = decision_tree.DecisionTreeClassifier()

    def test_fit(self):
        """ Check return value """
        X, y = load_iris(return_X_y=True)
        ret = self.model.fit(X[:5], y[:5])
        self.assertIsInstance(ret, decision_tree.DecisionTreeClassifier)

    def test_aggregate_target(self):
        """ Check return value """
        ret = self.model._aggregate_target(np.array([0, 1, 2, 1]))
        self.assertEqual(ret, 1)

    def test_calc_info_gain_entropy(self):
        """ criterion = 'entropy' """
        self.model.criterion = 'entropy'
        y = np.arange(10)
        left_y = y[:5]
        right_y = y[5:]
        gain = self.model._calc_info_gain(y, left_y, right_y)
        self.assertIsInstance(gain, np.float64)

    def test_calc_info_gain_gini(self):
        """ criterion = 'gini' """
        self.model.criterion = 'gini'
        y = np.arange(10)
        left_y = y[:5]
        right_y = y[5:]
        gain = self.model._calc_info_gain(y, left_y, right_y)
        self.assertIsInstance(gain, np.float64)

    def test_predict(self):
        X, y = load_iris(return_X_y=True)
        self.model.fit(X, y)
        y_pred = self.model.predict(X[:10])
        self.assertEqual(y_pred.shape, (10, ))
        self.assertTrue((y[:10] == y_pred[:10]).all())
