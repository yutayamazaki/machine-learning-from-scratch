import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))
