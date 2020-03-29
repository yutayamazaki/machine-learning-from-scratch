import numpy as np
from sklearn.tree import DecisionTreeClassifier


class AdaBoost(object):

    def __init__(self, num_estimators: int = 50, lr: float = 1.0):
        self.num_estimators = num_estimators
        self.lr = lr
        self.models = []
        self.alphas = []

    def fit(self, X, y):
        data_size = len(X)
        self.weights = np.full(data_size, 1 / len(X))
        unique_classes = np.unique(y)

        for i in range(self.num_estimators):
            model = DecisionTreeClassifier()
            model.fit(X, y, sample_weight=self.weights)
            y_pred = model.predict(X)
            error = self._calc_error_on_weak_classifier(X, y, y_pred, self.weights)
            alpha = self._calc_alpha_of_weak_classifier(error)
            self.weights = self._update_weights(
                self.weights,
                alpha,
                y,
                y_pred,
                self.lr
            )
            self.weights /= self.weights.sum()
            self.models.append(model)
            self.alphas.append(alpha)
            if error <= 0.0:
                return self
        return self

    def predict(self, X, y=None):
        y_pred = np.zeros_like(X[:, 0])
        num_estimators = len(self.models)
        for i in range(num_estimators):
            y_pred += self.alphas[i] * self.models[i].predict(X)
        return np.sign(y_pred)

    @staticmethod
    def _calc_error_on_weak_classifier(X, y_true, y_pred, weights):
        incorrect = y_pred != y_true
        return np.mean(np.average(incorrect, weights=weights, axis=0))

    @staticmethod
    def _calc_alpha_of_weak_classifier(error: float):
        if error == 0.:
            return 1e5
        alpha = np.log1p((1 - error) / error)
        return alpha

    @staticmethod
    def _update_weights(weights, alpha, y_true, y_pred, lr):
        incorrect = y_pred != y_true
        return -lr * weights * np.exp(alpha * incorrect * ((weights > 0) | (alpha < 0)))


if __name__ == '__main__':
    from sklearn.datasets import load_breast_cancer
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    model = AdaBoost(num_estimators=50)
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, shuffle=True, random_state=27)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    acc = accuracy_score(y_valid, y_pred)
    print(acc)

    model = DecisionTreeClassifier()
    y_pred = model.fit(X_train, y_train).predict(X_valid)

    acc = accuracy_score(y_valid, y_pred)
    print(acc)

    from sklearn.ensemble import AdaBoostClassifier
    model = AdaBoostClassifier()
    y_pred = model.fit(X_train, y_train).predict(X_valid)

    acc = accuracy_score(y_valid, y_pred)
    print(acc)
