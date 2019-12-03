import unittest

import numpy as np

from mlscratch import activations


class TestSigmoid(unittest.TestCase):

    def test_return(self):
        result = activations.sigmoid(np.array([0, 1]))

        self.assertIsInstance(result, np.ndarray)
        self.assertAlmostEqual(result[0], 0.5)


class TestReLU(unittest.TestCase):

    def test_return(self):
        x = np.array([-0.1, 0.0, 0.1])
        expected = np.array([0, 0, 1])

        result = activations.relu(x)

        self.assertIsInstance(result, np.ndarray)
        for i in range(3):
            self.assertAlmostEqual(result[i], expected[i])
