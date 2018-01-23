import matplotlib.pyplot as plt
import numpy as np


def normalize(X):
    return (X - X.mean(0)) / X.std(0)


def add_ones_col(X):
    ones = np.ones((X.shape[0], 1))
    return np.append(ones, X, axis=1)


def posteriori_probability(x, lambdas, K):
    den = 0
    lambda_dot_x = []
    for i in range(K):
        tmp = np.exp(lambdas[i].dot(x))
        den += tmp
        lambda_dot_x.append(tmp)
    return den, lambda_dot_x


def show_results_chart(data, Y_original, Y_kmeans, Y_rim, final_score):
    f, a = plt.subplots(nrows=3, ncols=1)
    f.suptitle('Final score: {}, \nGroups amount: {}'.format(final_score, len(set(Y_rim))))
    plt.subplot(2, 2, 1)
    plt.title('Original classification')
    plt.scatter(data[:, 1], data[:, 2], c=Y_original)
    plt.subplot(2, 2, 2)
    plt.title('KMeans classification')
    plt.scatter(data[:, 1], data[:, 2], c=Y_kmeans)
    plt.subplot(2, 2, 3)
    plt.title('RIM classification')
    plt.scatter(data[:, 1], data[:, 2], c=Y_rim)
    plt.show()


def assign_clusters(lambdas, X, K, N):
    y = []

    for i in range(N):
        den, cou = posteriori_probability(X[i], lambdas, K)
        cou /= den
        y.append(np.array(cou).argmax())
    return y
