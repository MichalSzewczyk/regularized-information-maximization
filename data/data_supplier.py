import numpy as np


class InputDataSupplier:
    def supply(self):
        return AlgorithmData(np.array([[1, 11, 3],
                         [1, 22, 3],
                         [1, 33, 3], ]), 2)


class AlgorithmData:
    def __init__(self, X, K):
        self.X = X
        self.K = K
