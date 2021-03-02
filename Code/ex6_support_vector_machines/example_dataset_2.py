# author : Lee
# date   : 2021/2/17 22:20

from pylab import *
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn import svm


def gaussian_kernel(x1, x2, sigma):
    return np.exp(-(np.sum((x1 - x2) ** 2)) / (2 * (sigma ** 2)))


def decision_boundary(svc, df, plt):
    x1_max = df['x1'].max() + 0.02
    x1_min = df['x1'].min() - 0.02
    x2_max = df['x2'].max() + 0.01
    x2_min = df['x2'].min() - 0.01
    x1_range = np.linspace(x1_min, x1_max, 100)
    x2_range = np.linspace(x2_min, x2_max, 100)
    x1, x2 = np.meshgrid(x1_range, x2_range)
    x1 = x1.reshape(x1.size, 1)
    x2 = x2.reshape(x2.size, 1)
    pre_y = svc.predict(np.hstack((x1, x2)))
    plt.scatter(x1, x2, s=3, marker='s', c=pre_y, cmap='Set3')


data = sio.loadmat(r'./data/ex6data2.mat')  # type: dict
df = pd.DataFrame(np.hstack((data['X'], data['y'])), columns=['x1', 'x2', 'y'])

pos = df[df['y'] == 1]
neg = df[df['y'] == 0]

fig = plt.figure(figsize=(17, 9), dpi=80)

for i, g in enumerate([1, 2, 4, 10, 16, 30]):
    svc = svm.SVC(C=100, kernel='rbf', gamma=g)
    svc.fit(df[['x1', 'x2']], df['y'])
    print("training accuracy(gamma = %s)\t:%f" % (svc.gamma, svc.score(df[['x1', 'x2']], df['y'])))
    ax = fig.add_subplot(2, 3, i + 1)  # 添加 2 * 3 个子图，子图索引按行从 1 开始
    decision_boundary(svc, df, ax)
    ax.scatter(pos['x1'], pos['x2'], c='c', s=25, marker='+', label='positive')
    ax.scatter(neg['x1'], neg['x2'], c='orange', s=15, marker='o', label='negative')
    ax.set_xlabel(r'$x_1$', fontsize=12)
    ax.set_ylabel(r'$x_2$', fontsize=12)
    ax.legend()
    ax.set_title(r"SVM with Gaussian Kernel ($\gamma$=%s)" % svc.gamma)

subplots_adjust(hspace=0.3)  # 调整子图横向间距
plt.show()
