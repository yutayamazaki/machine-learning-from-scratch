import sys

import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

sys.path.append('..')
from mlscratch.models.linear import LinearRegression
from mlscratch.preprocessing import StandardScaler
from mlscratch.models.losses import mean_squared_error

if __name__ == '__main__':
    model = LinearRegression()
    X, y = load_boston(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        shuffle=True,
        random_state=27
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)

    mse = mean_squared_error(y_valid, y_pred)
    print(f'Mean Squared Error: {mse}')

    plt.plot(model.train_metric_list)
    plt.title('Training MSE')
    plt.xlabel('Iterations')
    plt.ylabel('Mean Squared Error')
    plt.show()
