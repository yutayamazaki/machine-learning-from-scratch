import numpy as np


def binary_cross_entropy(y_true, y_pred, eps=1e-8):
    error = - y_true * np.log(y_pred + eps) - \
        (1 - y_true) * np.log(1 - y_pred + eps)
    return error.mean()


def mean_squared_error(y_true, y_pred):
    return np.power(y_true - y_pred, 2).mean()
