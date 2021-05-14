# author : Lee
# date   : 2021/3/1 19:45

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio


def pre_process(X):
    """
    数据预处理: 特征缩放(feature scaling)
    :param X:
    :return:
    """
    n = X.shape[1]
    for j in range(n):
        X[:, j] = (X[:, j] - X[:, j].mean()) / X[:, j].std()
    return X


def pca(X):
    """
    主成分分析
    :param X: (m, n)
    :return: u, s, vt
    """
    m = X.shape[0]
    sigma = (X.T @ X) / m
    u, s, vt = np.linalg.svd(sigma)
    return u, s, vt


def project_data(X, u, k):
    """
    映射数据到主成分上，样本特征从 n 维降维至 k 维
    :param X: (m, n)
    :param u: (n, n)
    :param k: 主成分数量
    :return: z (m, k)
    """
    u_reduce = u[:, :k]
    z = X @ u_reduce
    return z


def recover_data(z, u, k):
    """
    数据解压
    :param z: (m, k)
    :param u: (n, n)
    :param k: 主成分数量
    :return: X_approx: (m, n)
    """
    u_reduce = u[:, :k]  # (n, k)
    X_approx = z @ u_reduce.T
    return X_approx


if __name__ == '__main__':
    data = sio.loadmat(r'./data/ex7data1')
    X = data['X']
    X = pre_process(X)
    k = 1
    u, s, vt = pca(X)
    z = project_data(X, u, k)
    X_approx = recover_data(z, u, k)
    plt.figure(figsize=(6, 6), dpi=80)
    plt.scatter(X[:, 0], X[:, 1], marker='o', s=6, c='b')
    plt.scatter(X_approx[:, 0], X_approx[:, 1], marker='o', s=6, c='r')
    for i in range(X.shape[0]):
        plt.plot([X[i][0], X_approx[i][0]], [X[i][1], X_approx[i][1]], lw=0.5, c='k', ls='-.')
    plt.show()
