# author : Lee
# date   : 2021/2/18 21:44

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn import svm


def decision_boundary(svc, df, plt):
    x1_max = df['x1'].max()
    x1_min = df['x1'].min()
    x2_max = df['x2'].max()
    x2_min = df['x2'].min()
    x1_ex = (x1_max - x1_min) / 50
    x2_ex = (x2_max - x2_min) / 50
    x1_max += x1_ex
    x1_min -= x1_ex
    x2_max += x2_ex
    x2_min -= x2_ex
    x1_range = np.linspace(x1_min, x1_max, 200)
    x2_range = np.linspace(x2_min, x2_max, 200)
    x1, x2 = np.meshgrid(x1_range, x2_range)
    x1 = x1.reshape(x1.size, 1)
    x2 = x2.reshape(x2.size, 1)
    pre_y = svc.predict(np.hstack((x1, x2)))
    plt.scatter(x1, x2, s=8, marker='s', c=pre_y, cmap='Set3')


data = sio.loadmat(r'./data/ex6data3.mat')  # type: dict
train_df = pd.DataFrame(np.hstack((data['X'], data['y'])), columns=['x1', 'x2', 'y'])
cv_df = pd.DataFrame(np.hstack((data['Xval'], data['yval'])), columns=['x1', 'x2', 'y'])

C_val = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
gamma_val = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]

eva_df = pd.DataFrame(columns=['C', 'gamma', 'cv_accuracy'])

for c in C_val:
    for g in gamma_val:
        svc = svm.SVC(C=c, kernel='rbf', gamma=g)
        svc.fit(train_df[['x1', 'x2']], train_df['y'])
        score = svc.score(cv_df[['x1', 'x2']], cv_df['y'])
        eva_df = eva_df.append({'C': c, 'gamma': g, 'cv_accuracy': score}, ignore_index=True)

print(eva_df)
best_df = eva_df.loc[eva_df['cv_accuracy'].idxmax()]
print("-" * 4 + ' best parameter ' + '-' * 4 + '\n', best_df)
svc = svm.SVC(C=float(best_df['C']), kernel='rbf', gamma=float(best_df['gamma']))
svc.fit(train_df[['x1', 'x2']], train_df['y'])
score = svc.score(cv_df[['x1', 'x2']], cv_df['y'])

plt.figure(figsize=(7, 7), dpi=80)
decision_boundary(svc, train_df, plt)
pos = train_df[train_df['y'] == 1]
neg = train_df[train_df['y'] == 0]
plt.scatter(pos['x1'], pos['x2'], c='c', s=50, marker='+', label='positive')
plt.scatter(neg['x1'], neg['x2'], c='orange', s=30, marker='o', label='negative')
plt.xlabel(r'$x_1$', fontsize=12)
plt.ylabel(r'$x_2$', fontsize=12)
plt.title(r"SVM with Gaussian Kernel (C=%s ,$\gamma$=%s)" % (svc.C, svc.gamma))
plt.legend()
plt.show()
