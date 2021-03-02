# author : Lee
# date   : 2021/2/17 18:18

from pylab import *
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn import svm

data = sio.loadmat(r'./data/ex6data1.mat')  # type: dict
df = pd.DataFrame(np.hstack((data['X'], data['y'])), columns=['x1', 'x2', 'y'])

pos = df[df['y'] == 1]
neg = df[df['y'] == 0]

plt.figure(figsize=(12, 5), dpi=80)
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)

subplots_adjust(wspace=0.5)  # 调整子图横向间距

ax1.scatter(pos['x1'], pos['x2'], s=50, marker='+', label='positive')
ax1.scatter(neg['x1'], neg['x2'], s=30, marker='o', label='negative')
ax1.set_xlabel(r'$x_1$', fontsize=12)
ax1.set_ylabel(r'$x_2$', fontsize=12)

ax2.scatter(pos['x1'], pos['x2'], s=50, marker='+', label='positive')
ax2.scatter(neg['x1'], neg['x2'], s=30, marker='o', label='negative')
ax2.set_xlabel(r'$x_1$', fontsize=12)
ax2.set_ylabel(r'$x_2$', fontsize=12)

svc1 = svm.LinearSVC(C=1, max_iter=50000)
svc2 = svm.LinearSVC(C=100, max_iter=50000)
svc1.fit(df[['x1', 'x2']], df['y'])
svc2.fit(df[['x1', 'x2']], df['y'])

print("Accuracy with C = 1   :", svc1.score(df[['x1', 'x2']], df['y']))
print("Accuracy with C = 100 :", svc2.score(df[['x1', 'x2']], df['y']))

# decision boundary:
#   x1 * svc1.coef_[0] + x2 * svc2.coef_[1] + svc1.intercept_ = 0
x1 = np.linspace(0, 4, 100)
x2_1 = -(svc1.intercept_ + x1 * svc1.coef_[0][0]) / svc1.coef_[0][1]
x2_2 = -(svc2.intercept_ + x1 * svc2.coef_[0][0]) / svc2.coef_[0][1]

ax1.plot(x1, x2_1, lw=0.5, c='r')
ax2.plot(x1, x2_2, lw=0.5, c='r')

ax1.set_title(r'SVM Decision Boundary with $C=1$')
ax2.set_title(r'SVM Decision Boundary with $C=100$')
ax1.legend()
ax2.legend()

plt.show()
