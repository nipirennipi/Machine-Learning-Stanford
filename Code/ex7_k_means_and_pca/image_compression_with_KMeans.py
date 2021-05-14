# author : Lee
# date   : 2021/3/1 19:18

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import io

img = io.imread(r'./data/bird_small.png') / 255
X = img.reshape((img.shape[0] * img.shape[1]), img.shape[2])
K = 16
model = KMeans(n_clusters=K, n_init=10, max_iter=10)
model.fit(X)
centroids = model.cluster_centers_
l = model.predict(X)
compressed_img = centroids[l]
compressed_img = compressed_img.reshape((img.shape[0], img.shape[1], img.shape[2]))

plt.figure(figsize=(12, 5), dpi=80)
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)
ax1.imshow(img)
ax2.imshow(compressed_img)
plt.show()
