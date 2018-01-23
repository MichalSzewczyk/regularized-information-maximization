import numpy as np
import scipy.optimize as sc

from utils.algorithm_utils import posteriori_probability


def get_rim_lambdas(X, Y, K, D, N, alfa, lambdas):
    return np.reshape(sc.minimize(rim_for_matrix, lambdas, method='L-BFGS-B', args=(X, Y, K, D, N, alfa), jac=True).x,
                      (K, D))


def rim_for_matrix(lambdas_, X, Y, K, D, N, alfa):
    lambdas = lambdas_.reshape((K, D))
    jac = np.zeros((K, D))
    value = 0
    prob_matrix = get_prob_matrix(X, K, N, lambdas)
    mean_column_prob_matrix = prob_matrix.mean(0)
    sum = get_array_sum(K, N, prob_matrix, mean_column_prob_matrix)
    for i in range(N):
        value_, jac_ = rim_for_vector(X[Y[i]], i, Y[i], K, D, prob_matrix, mean_column_prob_matrix, sum)
        value += np.log(value_)
        jac += jac_
    r = np.sum(lambdas[:, :1] ** 2)
    value -= alfa * r
    jac[:, :1] -= 2 * lambdas[:, :1] * alfa
    jac /= N
    return -value, -jac.flatten()


def rim_for_vector(var, n, k, K, D, prob_matrix, mean_prob_matrix_column, sum):
    result = prob_matrix[n][k] * np.log(prob_matrix[n][k])
    jac = np.zeros((K, D))
    for k in range(K):
        result -= prob_matrix[n][k] * np.log(mean_prob_matrix_column[k])

    for i in range(K):
        for j in range(D):
            jac[i][j] = var[j] * prob_matrix[n][i] * (np.log(prob_matrix[n][i] / mean_prob_matrix_column[i]) - sum[n])

    return result, jac


def get_prob_matrix(X, K, N, lambdas):
    matrix = np.zeros((N, K))
    for i in range(N):
        for j in range(K):
            den, tmp = posteriori_probability(X[i], lambdas, K)
            matrix[i][j] = tmp[j] / den
    return matrix


def get_array_sum(K, N, prob_matrix, mean_column_prob_matrix):
    t = []
    for i in range(N):
        tmp = 0
        for j in range(K):
            tmp += prob_matrix[i][j] * np.log(prob_matrix[i][j] / mean_column_prob_matrix[j])
        t.append(tmp)
    return t
