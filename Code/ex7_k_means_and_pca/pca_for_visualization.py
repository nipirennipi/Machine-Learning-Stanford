# author : Lee
# date   : 2021/3/2 18:46

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import io
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

img = io.imread(r'./data/bird_small.png') / 255
X = img.reshape((img.shape[0] * img.shape[1], img.shape[2]))
K = 16
model = KMeans(n_clusters=K, n_init=10, max_iter=30)
model.fit(X)
centroids = model.cluster_centers_
l = model.predict(X)

pca = PCA(n_components=2)
z = pca.fit_transform(X)
X_approx = pca.inverse_transform(z)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=0.1, marker='o', c=l, cmap='rainbow')

plt.figure()
plt.scatter(z[:, 0], z[:, 1], s=0.1, marker='o', c=l, cmap='rainbow')

plt.show()
