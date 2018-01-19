from abc import abstractmethod
from sklearn.cluster import KMeans

import numpy as np


class Command:
    @abstractmethod
    def execute(self, input_data):
        pass


class NormalizingCommand(Command):
    def __init__(self, logger):
        self.logger = logger

    def execute(self, input_data):
        self.logger.info('Input data for algorithm: \n{}'.format(input_data.X))
        means_by_rows = np.mean(input_data.X, axis=1)
        std_by_rows = np.std(input_data.X, axis=1)
        input_data.X = (input_data.X - means_by_rows) / std_by_rows
        self.logger.info('Data after normalization: \n{}'.format(input_data.X))

        return input_data


class KMeansCommand(Command):
    def __init__(self, logger):
        self.logger = logger

    def execute(self, input_data):
        self.logger.info('Normalized input data: \n{}'.format(input_data.X))
        result = KMeans(n_clusters=input_data.K, init="k-means++").fit(input_data.X)
        self.logger.info('Labels of clusters: \n{}'.format(result.labels_))

        return result