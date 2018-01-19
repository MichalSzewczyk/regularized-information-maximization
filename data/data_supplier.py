import numpy as np


class InputDataSupplier:
    def getData(self, path, sep, name=False):
        with open(path) as file:
            tmp = [line.strip().split(sep) for line in file]
            data = []
        if name:
            for line in tmp:
                data.append(list(filter(lambda x: len(x), line))[1:-1])
        else:
            for line in tmp:
                data.append(list(filter(lambda x: len(x), line))[:-1])
        return data

    def supply(self):
        return AlgorithmData(np.matrix(np.array(self.getData('data/lenses.txt', ' ', name=True))).astype(np.float), 3)


class AlgorithmData:
    def __init__(self, X, K):
        self.X = X
        self.K = K
