import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


class ClassificationDTO:
    def __init__(self, X, alpha):
        self.X = X
        self.alpha = alpha


def load_data(data_source):
    user_knowledge = pd.read_csv(data_source).values
    return user_knowledge


def analyze(X, Y, alpha, k):
    pass


def normalize(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)


def add_constant_term(X):
    return np.hstack((np.ones((X.shape[0], 1)), X))


def kmeans_cluster(X, k):
    kmeans = KMeans(n_clusters=k)
    return kmeans.fit_predict(X)


k = 0
loaded_data = load_data('sample_data.csv')
examples = loaded_data[:, :-1].astype(float)
real_classification = loaded_data[:, -1]
examples = normalize(examples)
examples = add_constant_term(examples)
kmeans_labels = kmeans_cluster(examples, k)
kmeans_classes = len(set(kmeans_labels))

alphas = [0.0, 0.0001, 0.001, 0.01, 0.1, 1]

for alpha in alphas:
    analyze(examples, kmeans_labels, alpha, k)
