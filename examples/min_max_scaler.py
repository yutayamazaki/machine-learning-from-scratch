import sys

import numpy as np

sys.path.append('../')
from mlscratch.preprocessing import MinMaxScaler


if __name__ == "__main__":
    mat = np.array([
        [0, 1, 2],
        [5, 0, 1]
    ])
    scaled = MinMaxScaler().fit_transform(mat)
    print(scaled)
