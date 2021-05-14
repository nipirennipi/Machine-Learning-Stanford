# author : Lee
# date   : 2021/3/10 16:16

from anomaly_detection import *

data = sio.loadmat(r'.\data\ex8data2.mat')
X = data['X']
Xval = data['Xval']
yval = data['yval']

select_threshold(Xval, yval)
best_epsilon, best_f1_score = select_threshold(Xval, yval)
print("best_epsilon:\t", best_epsilon)
ypre = predict(X, best_epsilon)
anomaly = X[np.argwhere(ypre == 1).flatten(), :]
print("number of anomaly:\t", anomaly.shape[0])
