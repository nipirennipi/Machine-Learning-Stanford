# author : Lee
# date   : 2021/2/6 17:31

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import scipy.optimize as opt


def load_data(path):
    data = sio.loadmat(path)  # type: dict
    X = data["X"]
    y = data["y"]
    return X, y


def load_weight(path):
    weight = sio.loadmat(path)
    return weight["Theta1"], weight["Theta2"]


def display_data(X):
    """
    在训练集中随机抽取 100 张图显示
    :param X: 特征 (5000, 400)
    """
    size = int(np.sqrt(X.shape[1]))
    index = np.random.randint(0, X.shape[0], 100)
    images = X[index, :]  # (100, 400)
    fig, axes = plt.subplots(nrows=10, ncols=10, sharex="all", sharey="all", figsize=(8, 8))
    for r in range(10):
        for c in range(10):
            axes[r, c].matshow(images[10 * r + c].reshape(size, size).T, cmap=matplotlib.cm.binary)
            plt.xticks([])
            plt.yticks([])
    plt.show()


def recode_label(y):
    """
    recode the labels as vectors containing only values 0 or 1
    y[i] = 10 : label[i] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1].T
    :param y:
    :return:
    """
    labels = []
    for i in y:
        label = np.zeros(10, dtype="int")
        label[i - 1] = 1
        labels.append(label)
    return np.array(labels).T


def unroll(theta1, theta2):
    """
    unroll into a vector
    """
    return np.concatenate((theta1.flatten(), theta2.flatten()))


def roll(theta):
    """
    将参数向量转换为权重矩阵
    神经网络的架构:(400 + 1) * (25 + 1) * 10
    :param theta: 所有参数展开为一个向量
    :return: theta1, theta2
    """
    theta1 = theta[:25 * 401].reshape(25, 401)
    theta2 = theta[25 * 401:].reshape(10, 26)
    return theta1, theta2


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_gradient(z):
    a = sigmoid(z)
    return a * (1 - a)


def cost_function(theta, X, y, regularized=False, l=1):
    """
    don't regularize the terms that correspond to the bias.
    :param theta: 所有参数展开为一个向量
    :param X: 特征 (401, 5000)
    :param y: 标签 (10, 5000)
    :param regularized: 正则化
    :param l: 正则化参数lambda
    :return:
    """
    m = y.shape[1]
    a1, z2, a2, z3, h = feed_forward(theta, X)
    cost = -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)) / m
    if regularized:
        theta1, theta2 = roll(theta)  # (25, 401), (10, 26)
        regularized_term = l / (2 * m) * (np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2)))
        cost += regularized_term
    return cost


def feed_forward(theta, X):
    """
    :param theta: 所有参数展开为一个向量
    :param X: 特征 (401, 5000)
    :return:
    """
    theta1, theta2 = roll(theta)  # (25, 401), (10, 26)
    a1 = X  # (401, 5000)
    z2 = theta1 @ X  # (25, 5000)
    a2 = np.insert(sigmoid(z2), 0, np.ones(z2.shape[1]), axis=0)  # (26, 5000)
    z3 = theta2 @ a2  # (10, 5000)
    h = sigmoid(z3)  # (10, 5000)
    return a1, z2, a2, z3, h


def random_initialization(size, epsilon=0.12):
    """
    initialize each parameter to a random value in [-epsilon,+epsilon]. (actually [-epsilon,epsilon) )
    :param epsilon:
    :return:
    """
    return np.random.uniform(-epsilon, epsilon, size)


def back_propagation(theta, X, y, regularized=False, l=1):
    """
    :param theta: 所有参数展开为一个向量
    :param X: 特征 (401, 5000)
    :param y: 标签 (10, 5000)
    :param regularized: 正则化
    :param l: 正则化参数lambda
    :return:
    """
    m = X.shape[1]
    theta1, theta2 = roll(theta)  # (25, 401), (10, 26)
    Delta1 = np.zeros(theta1.shape)  # (25, 401)
    Delta2 = np.zeros(theta2.shape)  # (10, 26)
    # (401, 5000), (25, 5000), (26, 5000), (10, 5000), (10, 5000)
    a1, z2, a2, z3, h = feed_forward(theta, X)
    for i in range(m):
        delta3 = (h[:, i] - y[:, i])[:, None]  # (10, 1)
        delta2 = theta2.T @ delta3 * sigmoid_gradient(np.insert(z2[:, i], 0, 1, axis=0)[:, None])  # (26, 1)
        Delta1 += delta2[1:, :] @ a1[:, i][:, None].T  # (25, 401)
        Delta2 += delta3 @ a2[:, i][:, None].T  # (10, 26)
    D1 = Delta1 / m
    D2 = Delta2 / m
    if regularized:
        D1[:, 1:] = D1[:, 1:] + (l / m) * theta1[:, 1:]
        D2[:, 1:] = D2[:, 1:] + (l / m) * theta2[:, 1:]
    return unroll(D1, D2)


def gradient_checking(theta, X, y, regularized=False, epsilon=0.0001):
    """
    when the number of examples is 5000:
    regularized=False: relative difference: 2.1391810824853674e-09
    regularized=True : relative difference: 3.172015041999083e-09
    :param theta: 所有参数展开为一个向量
    :param X: 特征 (401, 5000)
    :param y: 标签 (10, 5000)
    :param regularized: 正则化
    :param epsilon: 双侧差分步长
    :return:
    """
    D = back_propagation(theta, X, y, regularized)  # gradient that computed using back propagation
    l = []
    for i in range(len(theta)):
        theta_plus = theta.copy()
        theta_minus = theta.copy()
        theta_plus[i] += epsilon
        theta_minus[i] -= epsilon
        if regularized:
            l.append(
                (cost_function(theta_plus, X, y, regularized=True) - cost_function(theta_minus, X, y, regularized=True))
                / (2 * epsilon))
        else:
            l.append((cost_function(theta_plus, X, y) - cost_function(theta_minus, X, y)) / (2 * epsilon))
    D_ = np.array(l)  # gradient that computed using numerical estimate
    relative_difference = np.linalg.norm(D - D_) / np.linalg.norm(D + D_)
    print("If your backpropagation implementation is correct,\n"
          "you should see a relative difference that is less than 1e-9.(epsilon=%f)" % epsilon)
    print("relative difference:", relative_difference)


def split_data(X, y, k=0.7):
    """
    按照比例 k 将数据集分为训练集与测试集，且同一标签的数据划分比例也是 k
    :param X: 特征 (5000, 400)
    :param y: 标签 (5000, 1)
    :param k: 训练集在数据集中占比
    :return:
    """
    n = int(X.shape[0] / 10)
    train_X = np.array([]).reshape(0, X.shape[1])
    train_y = np.array([], dtype="int").reshape(0, y.shape[1])
    test_X = np.array([]).reshape(0, X.shape[1])
    test_y = np.array([], dtype="int").reshape(0, y.shape[1])
    for i in range(10):
        train_X = np.insert(train_X, train_X.shape[0], X[i * n:int((i + k) * n), :], axis=0)
        train_y = np.insert(train_y, train_y.shape[0], y[i * n:int((i + k) * n), :], axis=0)
        test_X = np.insert(test_X, test_X.shape[0], X[int((i + k) * n):(i + 1) * n:, :], axis=0)
        test_y = np.insert(test_y, test_y.shape[0], y[int((i + k) * n):(i + 1) * n:, :], axis=0)
    return train_X, train_y, test_X, test_y


def train(X, y):
    """
    :param X: 特征 (3500, 400)
    :param y: 标签 (3500, 1)
    :return:
    """
    theta1, theta2 = load_weight("ex4weights.mat")  # (25, 401),(10, 26)
    X = np.insert(X, 0, np.ones(X.shape[0]), axis=1).T  # X:(401, 5000),add bias in the first row
    y = recode_label(y)  # y:(10,5000)
    """Gradient Checking:
    # parameters that correspond to the bias is in the first column
    theta = unroll(theta1, theta2)
    gradient_checking(theta, X[:, 1:2], y[:, 1:2], regularized=True)
    """
    init_theta = random_initialization(theta1.size + theta2.size)
    res = opt.minimize(fun=cost_function, x0=init_theta, args=(X, y, True, 1),
                       method='TNC', jac=back_propagation, options={'maxiter': 400})
    print(res)
    np.savetxt("theta.csv", res.x, delimiter=",")


def compute_accuracy(trained_theta, X, y):
    """
    training accuracy computed using test set
    :param trained_theta: trained parameters
    :param X: 特征 (1500, 400)
    :param y: 标签 (1500, 1)
    :return:
    """
    X = np.insert(X, 0, np.ones(X.shape[0]), axis=1).T  # X:(401, 5000),add bias in the first row
    m = X.shape[0]
    # h:(10, 5000)
    a1, z2, a2, z3, h = feed_forward(trained_theta, X)
    correct_count = 0
    for i in range(m):
        # if np.argmax(h[:, i]) + 1 == y[i]:
        if h[y[i] - 1, i] > 0.5:
            correct_count += 1
    print("accuracy:", correct_count / m)


def main():
    X, y = load_data("ex4data1.mat")  # (5000, 400),(5000,1)
    """Take 100 samples for visualization
    display_data(X)
    """
    train_X, train_y, test_X, test_y = split_data(X, y)
    # train(train_X, train_y)
    trained_theta = np.loadtxt(open(r".\theta.csv"), delimiter=",")
    compute_accuracy(trained_theta, test_X, test_y)


if __name__ == '__main__':
    main()
