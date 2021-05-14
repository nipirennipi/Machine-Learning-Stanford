# author : Lee
# date   : 2021/3/1 18:21

from k_means import *

data = sio.loadmat(r'./data/bird_small.mat')
img = data['A'] / 255  # (128, 128, 3)
X = img.reshape((img.shape[0] * img.shape[1]), img.shape[2])
K = 16
initial_centroids = random_init_centroids(X, K)
centroids, l = k_means(X, initial_centroids, 1)
l = l.astype(np.int)
compressed_img = centroids[l, :]
compressed_img = compressed_img.reshape((img.shape[0], img.shape[1], img.shape[2]))

plt.figure(figsize=(12, 5), dpi=80)
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)
ax1.imshow(img)
ax2.imshow(compressed_img)
plt.show()
