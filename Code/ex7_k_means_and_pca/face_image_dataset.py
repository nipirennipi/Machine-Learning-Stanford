# author : Lee
# date   : 2021/3/1 21:24


from pca import *


def display_data(X, title):
    """
    选择前 100 张人脸图片显示
    :param X: 人脸集 (5000, 1024)
    :return:
    """
    size = int(np.sqrt(X.shape[1]))
    # index = np.random.randint(0, X.shape[0], 100)
    index = np.arange(100)
    images = X[index, :]
    count = int(np.sqrt(len(index)))
    fig, axes = plt.subplots(nrows=count, ncols=count, sharex="all", sharey="all", figsize=(6, 6))
    plt.suptitle(title)
    for r in range(count):
        for c in range(count):
            axes[r, c].imshow(images[count * r + c].reshape(size, size).T)
            plt.xticks([])
            plt.yticks([])


data = sio.loadmat(r'./data/ex7faces.mat')
X = data['X']  # (5000, 1024)
display_data(X, 'Original images')

for k in [50, 100, 250]:
    X = pre_process(X)
    u, s, vt = pca(X)
    z = project_data(X, u, k)
    X_approx = recover_data(z, u, k)
    display_data(X_approx, 'recovered from $k=%d$' % k)
    print("k = %d   \t: %f" % (k, np.sum(s[:k]) / np.sum(s)))

plt.show()
