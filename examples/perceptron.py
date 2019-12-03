import sys

import numpy as np

sys.path.append('../')
from mlscratch.perceptron import Perceptron


if __name__ == "__main__":
    # AND
    X_train = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    y_train = np.array([0, 0, 0, 1])

    # collect answer is: [0, 0, 0, 1].
    model = Perceptron()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    print(y_pred)
