import sys
sys.path.append('../')

import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from mlscratch.models.decision_tree import DecisionTreeClassifier

if __name__ == '__main__':
    X, y = load_iris(return_X_y=True)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, shuffle=True, random_state=27
    )
    model = DecisionTreeClassifier(criterion='gini')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)

    acc = accuracy_score(y_pred, y_valid)
    print(f'Accuracy: {acc}')
