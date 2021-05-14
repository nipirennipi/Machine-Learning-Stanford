# author : Lee
# date   : 2021/3/9 19:21

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy import stats


def estimate_gaussian(X):
    """
    计算 X 中每个特征的均值与方差
    :param X: (m, n)
    :return: mu, Sigma: 均值向量, 协方差矩阵
    """
    mu = np.mean(X, axis=0)
    Sigma = np.cov(X.T)
    return mu, Sigma


def select_threshold(Xval, yval):
    """
    确定阈值
    :param Xval: (m, n)
    :param yval: (n,)
    :return:
    """
    mu, Sigma = estimate_gaussian(Xval)
    pval = stats.multivariate_normal.pdf(Xval, mean=mu, cov=Sigma)
    epsilon = np.linspace(pval.min(), pval.max(), 1000)
    f1_score = np.empty(epsilon.shape)
    for i, e in enumerate(epsilon):
        ypre = pval
        ypre = np.where(ypre < e, 1, 0).reshape((ypre.shape[0], 1))
        TP = np.sum(np.logical_and(yval == 0, ypre == 0))
        FP = np.sum(np.logical_and(yval == 1, ypre == 0))
        FN = np.sum(np.logical_and(yval == 0, ypre == 1))
        P = TP / (TP + FP)
        R = TP / (TP + FN)
        f1_score[i] = 2 * (P * R) / (P + R)
    best_arg = np.argmax(f1_score)
    best_epsilon = epsilon[best_arg]
    best_f1_score = f1_score[best_arg]
    return best_epsilon, best_f1_score


def predict(X, e):
    """
    对 X 进行异常检测
    :param X:
    :param e:
    :return:
    """
    mu, Sigma = estimate_gaussian(X)
    pval = stats.multivariate_normal.pdf(X, mean=mu, cov=Sigma)
    ypre = np.where(pval < e, 1, 0)
    return ypre


if __name__ == '__main__':
    data = sio.loadmat(r'./data/ex8data1.mat')
    X = data['X']
    Xval = data['Xval']
    yval = data['yval']

    plt.figure(figsize=(6, 6), dpi=80)
    plt.scatter(X[:, 0], X[:, 1], s=25, marker='+', label='normal')

    select_threshold(Xval, yval)
    best_epsilon, best_f1_score = select_threshold(Xval, yval)
    print("best_epsilon:\t", best_epsilon)
    ypre = predict(X, best_epsilon)
    anomaly = X[np.argwhere(ypre == 1).flatten(), :]
    plt.scatter(anomaly[:, 0], anomaly[:, 1], s=25, marker='+', c='r', label='anomaly')
    plt.legend()
    plt.show()
