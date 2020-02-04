import numpy as np


class _BaseTransformer:

    def fit(self, X, y=None):
        raise NotImplementedError()

    def transform(self, X, y=None):
        raise NotImplementedError()

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)


class PCA(_BaseTransformer):

    def __init__(self, num_components: int):
        self.num_components = num_components

    def fit(self, X, y=None):
        self.X_scaled_ = (X - X.mean()) / X.std()
        X_cov = np.cov(self.X_scaled_.T, bias=0)
        eig, eig_vec = np.linalg.eig(X_cov)

        indices = np.argsort(eig)[::-1]
        eig = eig[indices]
        self.eig_vec_ = eig_vec[:, indices]
        return self

    def transform(self, X, y=None):
        X_reduced = np.dot(
            self.X_scaled_,
            self.eig_vec_[:, :self.num_components]
        )
        return X_reduced
