import sys

import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

sys.path.append('..')
from mlscratch.models.logistic_regression import LogisticRegression
from mlscratch.metrics import accuracy_score

if __name__ == '__main__':
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model = LogisticRegression()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {acc}')

    plt.plot(model.train_metric_list)
    plt.title('Training BCE')
    plt.xlabel('Iterations')
    plt.ylabel('Binary Cross Entropy')
    plt.show()
