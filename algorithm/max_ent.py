import numpy as np
import scipy.optimize as sc

from utils.algorithm_utils import posteriori_probability


def get_max_ent_lambdas(X, Y, K, D, N, alpha):
    return sc.minimize(max_ent_fun_vector, np.random.normal(loc=0, scale=1, size=K * D), method='L-BFGS-B',
                       args=(X, Y, K, D, N, alpha), jac=True).x


def max_ent_fun_vector(lambdas, X, Y, K, D, N, alpha):
    lambdas = lambdas.reshape((K, D))
    jac = np.zeros((K, D))
    value = 0
    for i in range(N):
        value_, jac_ = max_ent_fun(X[i], Y[i], lambdas, K, D)
        value += np.log(value_)
        jac += jac_
    r = np.sum(lambdas[:, :1] ** 2)
    value -= alpha * r
    jac[:, :1] -= 2 * lambdas[:, :1] * alpha
    return -value, -jac.flatten()


def max_ent_fun(x, y, lambdas, K, D):
    jac = np.zeros((K, D))
    d, lambda_dot_x = posteriori_probability(x, lambdas, K)
    c = lambda_dot_x[y]

    for i in range(K):
        for j in range(D):
            if i == j:
                jac[i][j] = (x[j] * (d - c)) / d
            else:
                jac[i][j] = -(x[j] * lambda_dot_x[i]) / d

    return c / d, jac


