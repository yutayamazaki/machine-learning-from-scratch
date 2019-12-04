import sys
sys.path.append('../')

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mlscratch.models.k_neighbor import KNN
from mlscratch.metrics import accuracy_score

if __name__ == '__main__':
    X, y = load_iris(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)

    model = KNN()
    model.fit(X_train, y_train)
    y_pred_proba = model.predict(X_valid)
    y_pred = np.round(y_pred_proba)

    acc = accuracy_score(y_valid, y_pred)
    print(f'Accuracy: {acc}')
