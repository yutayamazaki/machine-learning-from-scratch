import sys

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

sys.path.append('..')
from mlscratch.models.multilayer_perceptron import MLP
from mlscratch.metrics import accuracy_score


if __name__ == '__main__':
    X, y = load_iris(return_X_y=True)
    encoder = OneHotEncoder()
    y = encoder.fit_transform(y.reshape(-1, 1)).toarray()
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, shuffle=True, random_state=27
    )

    model = MLP(num_hidden=64)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_valid)
    acc = accuracy_score(y_valid.argmax(axis=1), y_pred.argmax(axis=1))

    print(f'Accuracy: {acc}')
