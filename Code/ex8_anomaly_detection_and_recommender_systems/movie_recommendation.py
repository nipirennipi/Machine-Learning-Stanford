# author : Lee
# date   : 2021/3/11 17:47

import scipy.optimize as opt
from recommender_systems import *


def mean_normalization(Y, R):
    # n_m = Y.shape[0]
    # Y_mu = np.sum((Y * R), axis=1) / np.sum(R, axis=1)
    # Y_norm = np.zeros(Y.shape)
    # for i in range(n_m):
    #     idx = np.where(R[i, :] == 1)
    #     Y_norm[i, idx] = Y[i, idx] - Y_mu[i]
    # Y_mu = Y_mu.reshape(n_m, 1)
    Y_mu = Y.mean()
    Y_norm = Y - Y_mu
    return Y_mu, Y_norm


def train(Y, R, n):
    """
    :param Y: (n_m, n_u)
    :param R: (n_m, n_u)
    :param n: number of feature
    :return:
    """
    n_m = Y.shape[0]
    n_u = Y.shape[1]
    Y_mu, Y_norm = mean_normalization(Y, R)
    init_X = np.random.standard_normal((n_m, n))
    init_Theta = np.random.standard_normal((n_u, n))
    init_params = unroll(init_X, init_Theta)
    res = opt.minimize(fun=cofi_costfunc, x0=init_params, args=(Y_norm, R, n, True, 10),
                       method='TNC', jac=gradient, options={'maxiter': 100})
    print(res)
    np.savetxt("params.csv", res.x, delimiter=",")
    # Predicted rating matrix
    X = res.x[:n_m * n].reshape(n_m, n)
    Theta = res.x[n_m * n:].reshape(n_u, n)
    Y_pre = X @ Theta.T + Y_mu
    np.savetxt("Predicted rating matrix.csv", Y_pre, delimiter=",")


def main():
    data1 = sio.loadmat(r'./data/ex8_movies.mat')
    Y = data1['Y']
    R = data1['R']
    # index of movies
    movies = {}
    with open(r'./data/movie_ids.txt') as f:
        for movie in f:
            movie = movie.split(maxsplit=1)
            movies[int(movie[0])] = movie[1].strip()

    Y_me = np.zeros((Y.shape[0], 1))
    Y_me[[0, 6, 11, 53, 63, 65, 68, 97, 182, 225, 354]] = [[4], [3], [5], [4], [5], [3], [5], [2], [4], [5], [5]]
    R_me = np.where(Y_me == 0, 0, 1)
    Y = np.insert(Y, [0], Y_me, axis=1)
    R = np.insert(R, [0], R_me, axis=1)
    n = 10
    Y_mu, Y_norm = mean_normalization(Y, R)
    # train(Y, R, n)

    Y_pre = np.loadtxt(open(r".\Predicted rating matrix.csv"), delimiter=",")
    Y_pre_me = Y_pre[:, 0]
    idx = np.argsort(Y_pre_me)[::-1]
    print("Top recommendations for you:")
    for i in idx[:20]:
        print("Predicting rating %.1f for %s" % (Y_pre_me[i], movies[i + 1]))


if __name__ == '__main__':
    main()
