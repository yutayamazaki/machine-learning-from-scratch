import numpy as np
from sklearn.tree import DecisionTreeClassifier


def bootstrap_sampling(X, y=None, sample_size: float=1.0):
    data_size = len(X)
    num_samples = int(data_size * sample_size)

    sample_indices = np.random.choice(
        np.arange(data_size),
        size=num_samples,
        replace=True
    )
    if y is None:
        return X[sample_indices]
    return X[sample_indices], y[sample_indices]



class Bagging(object):

    def __init__(self, num_estimators: int = 50, sample_size: float = 1.0):
        self.num_estimators = num_estimators
        self.sample_size = sample_size
        self.estimators = []

    def fit(self, X, y):
        for i in range(self.num_estimators):
            X_sample, y_sample = bootstrap_sampling(X, y, self.sample_size)
            estimator = DecisionTreeClassifier()
            estimator.fit(X_sample, y_sample)
            self.estimators.append(estimator)
        return self

    def predict(self, X):
        num_samples = len(X)
        y_pred = np.zeros(num_samples)
        for estimator in self.estimators:
            y_pred += estimator.predict(X)
        return np.round(y_pred / len(self.estimators))
