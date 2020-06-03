from typing import List

import numpy as np
from sklearn.tree import DecisionTreeRegressor


class CrossEntropy:

    def __call__(self, y_true, y_pred):
        pred = y_pred
        grad = pred - y_true
        return grad
        # hess = pred * (1. - pred)
        # return grad, hess


class GradientBoostingClassifier:

    def __init__(self, num_estimators: int = 100, lr: float = 0.1):
        self.num_estimators: int = num_estimators
        self.lr: float = lr
        self.objective = CrossEntropy()
        self.y_init = None

    def fit(self, X, y):
        self.y_init = np.mean(y)

        self.estimators: List[object] = []
        y_pred = np.full_like(y, self.y_init).astype('float32')
        for idx in range(self.num_estimators):
            grad = self.objective(y, y_pred)
            estimator = DecisionTreeRegressor()
            estimator.fit(X, grad)
            update = estimator.predict(X)
            y_pred -= np.multiply(self.lr, update)
            self.estimators.append(estimator)

        return self

    def predict(self, X):
        y_pred = np.full(len(X), self.y_init).astype('float32')
        for i in range(self.num_estimators):
            update = self.estimators[i].predict(X)
            update = np.multiply(self.lr, update)
            y_pred = y_pred - update
        # return self.objective.activation(y_pred)
        return np.round(y_pred)


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    X, y = load_iris(return_X_y=True)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)

    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)

    acc = accuracy_score(y_pred, y_valid)
    print(acc)

    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    acc = accuracy_score(y_pred, y_valid)
    print(acc)
