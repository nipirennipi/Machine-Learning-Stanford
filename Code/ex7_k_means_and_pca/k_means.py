# author : Lee
# date   : 2021/2/28 19:35

import random
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


def find_closest_centroids(X, centroids):
    """
    簇分配(cluster assignment)
    :param X: 样本 (m, n)
    :param centroids: 聚类中心 (K, n)
    :return: l: l[i] is the index of the centroid that is closest to X[i,:]
    """
    m = X.shape[0]
    l = np.zeros(m)
    K = centroids.shape[0]
    for i in range(m):
        min_dist = np.sum((X[i] - centroids[0]) ** 2)
        for k in range(1, K):
            dist = np.sum((X[i] - centroids[k]) ** 2)
            if dist < min_dist:
                min_dist = dist
                l[i] = k
    return l


def compute_centroids(X, l):
    """
    更新聚类中心
    :param X: 样本 (m, n)
    :param l: l[i] is the index of the centroid that is closest to X[i,:]
    :return: centroids: 聚类中心 (K, n)
    """
    K = np.unique(l).astype(np.int)
    n = X.shape[1]
    centroids = np.empty((len(K), n))
    for k in K:
        a = np.sum(X[np.where(l == k)])
        centroids[k] = np.sum(X[np.where(l == k)], axis=0) / np.sum(l == k)
    return centroids


def k_means(X, initial_centroids, max_iters=100):
    """
    k-means
    :param X: 样本 (m, n)
    :param initial_centroids: 初始聚类中心
    :param max_iters: 最大迭代次数
    :return: l: l[i] is the index of the centroid that is closest to X[i,:]
    """
    m = X.shape[0]
    l = np.zeros(m)
    centroids = initial_centroids.copy()
    for it in range(max_iters):
        pre_centroids = centroids.copy()
        l = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, l)
        if (pre_centroids == centroids).all():
            break
    return centroids, l


def random_init_centroids(X, K):
    """
    从 X 中随机选择 K 个样本作为聚类中心
    :param X: 样本 (m, n)
    :param K: 簇数
    :return: initial_centroids: 随机初始化的聚类中心
    """
    m = X.shape[0]
    initial_centroids = X[random.sample(range(m), K), :]
    return initial_centroids


if __name__ == '__main__':
    data = sio.loadmat(r'./data/ex7data2.mat')
    X = data['X']
    K = 3
    # initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
    initial_centroids = random_init_centroids(X, K)
    plt.figure(figsize=(7, 5), dpi=80)
    centroids, l = k_means(X, initial_centroids, 10)
    plt.scatter(X[:, 0], X[:, 1], s=10, marker='o', c=l, cmap='rainbow')
    plt.show()
