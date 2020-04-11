import sys
sys.path.append('../')

import numpy as np
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from mlscratch.models.decision_tree import DecisionTreeRegressor

if __name__ == '__main__':
    X, y = load_boston(return_X_y=True)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, shuffle=True, random_state=27
    )
    model = DecisionTreeRegressor(criterion='mse')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)

    mse = mean_squared_error(y_pred, y_valid)
    print(f'MSE: {mse}')
