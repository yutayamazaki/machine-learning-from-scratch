import numpy as np


def binary_cross_entropy(y_true, y_pred, size_average=True, eps=1e-8):
    log_y = np.log(y_pred+eps)
    error = y_true.dot(log_y) + (1 - y_true).dot(log_y)
    return error.mean() if size_average else error.sum()
