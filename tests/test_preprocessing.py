import unittest

import numpy as np

from mlscratch.preprocessing import MinMaxScaler


class TestMinMaxScaler(unittest.TestCase):

    def setUp(self):
        self.X = np.array([
            [0, 1, 2],
            [10, 20, 30]
        ])

    def test_fit_return(self):
        """ Check MinMaxScaler.fit returns itself. """
        ret = MinMaxScaler().fit(self.X)
        self.assertIsInstance(ret, MinMaxScaler)

    def test_fit_attributes(self):
        """ Check new attributes after fitted. """
        ret = MinMaxScaler().fit(self.X)
        expected_xmin_ = np.array([0, 1, 2])
        self.assertAlmostEqual(ret.x_min_.all(), expected_xmin_.all())

        expected_xmax_ = np.array([10, 19, 28])
        self.assertAlmostEqual(ret.x_max_.all(), expected_xmax_.all())

    def test_transform(self):
        """ Check return value and its dtype. """
        ret = MinMaxScaler().fit(self.X).transform(self.X)
        self.assertIsInstance(ret, np.ndarray)

        expected = np.array([
            [0, 0, 0],
            [1, 1, 1]
        ])
        self.assertAlmostEqual(ret.all(), expected.all())

    def test_fit_transform(self):
        """ Check it returns same result as self.fit().transform(). """
        ret = MinMaxScaler().fit_transform(self.X)
        expected = MinMaxScaler().fit(self.X).transform(self.X)
        self.assertAlmostEqual(ret.all(), expected.all())
