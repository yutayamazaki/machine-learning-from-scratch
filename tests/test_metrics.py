import unittest

import numpy as np

from mlscratch import metrics


class TestAccuracyScore(unittest.TestCase):

    def setUp(self):
        self.y_true = np.array([0, 1, 2, 3])
        self.y_pred = np.array([0, 1, 2, 2])

    def test_return(self):
        acc = metrics.accuracy_score(self.y_true, self.y_pred)
        self.assertAlmostEqual(acc, 0.75)
        self.assertIsInstance(acc, np.float64)
