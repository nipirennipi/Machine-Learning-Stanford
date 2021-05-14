# author : Lee
# date   : 2021/3/11 15:13

import numpy as np
import scipy.io as sio


def unroll(X, Theta):
    """
    unroll into a vector
    """
    return np.concatenate((X.flatten(), Theta.flatten()))


def cofi_costfunc(params, Y, R, n, regularized=False, l=1.5):
    """
    The collaborative filtering cost function (without regularization)
    :param params: unroll the parameters (X, Theta) into a single vector
    :param Y: (n_m, n_u)
    :param R: (n_m, n_u)
    :param n: number of feature
    :param regularized: regularized or not
    :param l: regularization parameter lambda
    :return:
    """
    n_m = Y.shape[0]
    n_u = Y.shape[1]
    X = params[:n_m * n].reshape(n_m, n)
    Theta = params[n_m * n:].reshape(n_u, n)
    cost = np.sum(((X @ Theta.T - Y) * R) ** 2)
    if regularized:
        cost += l * (np.sum(Theta ** 2) + np.sum(X ** 2))
    return cost / 2


def gradient(params, Y, R, n, regularized=False, l=1.5):
    """
    gradient (without regularization).
    :param params: unroll the parameters (X, Theta) into a single vector
    :param Y: (n_m, n_u)
    :param R: (n_m, n_u)
    :param n: number of feature
    :param regularized: regularized or not
    :param l: regularization parameter lambda
    :return:
    """
    n_m = Y.shape[0]
    n_u = Y.shape[1]
    X = params[:n_m * n].reshape(n_m, n)
    Theta = params[n_m * n:].reshape(n_u, n)
    temp = (X @ Theta.T - Y) * R
    X_grad = temp @ Theta
    Theta_grad = temp.T @ X
    if regularized:
        X_grad += l * X
        Theta_grad += l * Theta
    return unroll(X_grad, Theta_grad)


if __name__ == '__main__':
    data1 = sio.loadmat(r'./data/ex8_movies.mat')
    Y = data1['Y']
    R = data1['R']

    data2 = sio.loadmat(r'./data/ex8_movieParams.mat')
    X = data2['X']
    Theta = data2['Theta']

    users = 4
    movies = 5
    features = 3

    X_sub = X[:movies, :features]
    Theta_sub = Theta[:users, :features]
    Y_sub = Y[:movies, :users]
    R_sub = R[:movies, :users]

    param_sub = unroll(X_sub, Theta_sub)
    print(cofi_costfunc(param_sub, Y_sub, R_sub, features, True))
    print(gradient(param_sub, Y_sub, R_sub, features, True))
