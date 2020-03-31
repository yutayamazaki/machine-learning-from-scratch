import sys

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

sys.path.append('..')
from mlscratch.models.bagging import Bagging
from mlscratch.metrics import accuracy_score

if __name__ == '__main__':
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, shuffle=True, random_state=27
    )

    model = Bagging(num_estimators=10, sample_size=1.0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)

    acc = accuracy_score(y_valid, y_pred)
    print('Accuracy on Bagging')
    print(f'\t{acc}')

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)

    acc = accuracy_score(y_valid, y_pred)
    print('Accuracy on DecisionTreeClassifier')
    print(f'\t{acc}')
