import unittest

import numpy as np
from sklearn.datasets import load_iris

from mlscratch.decomposition.pca import PCA


class PCATests(unittest.TestCase):

    X, y = load_iris(return_X_y=True)
    num_components = 2

    def test_fit(self):
        """ Check return PCA itself and new attributes """
        pca = PCA(num_components=self.num_components).fit(self.X)
        self.assertIsInstance(pca, PCA)

        self.assertIsInstance(pca.X_scaled_, np.ndarray)
        self.assertIsInstance(pca.eig_vec_, np.ndarray)

    def test_transform(self):
        """ Check dtype and shape of returned value """
        pca = PCA(num_components=self.num_components)
        X_reduced = pca.fit(self.X).transform(self.X)
        self.assertEqual(X_reduced.shape, (150, 2))
        self.assertIsInstance(X_reduced, np.ndarray)

    def test_fit_transform(self):
        pca = PCA(num_components=self.num_components)
        X_reduced = pca.fit_transform(self.X)
        self.assertEqual(X_reduced.shape, (150, 2))
        self.assertIsInstance(X_reduced, np.ndarray)
