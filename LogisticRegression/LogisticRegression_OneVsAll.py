# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
from scipy import optimize
from matplotlib.font_manager import FontProperties
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体 解决中文显示方块问题
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)  # 解决windows环境下画图汉字乱码问题


def logisticRegression_OneVsAll():
    data = loadmat_data("data_digits.mat")
    X = data['X']  # 获取X数据，每一行对应一个数字20x20px
    y = data['y']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    num_labels = 10  # 数字个数，0-9

    ##　随机显示几行数据
    # m, n = x_train.shape
    # rand_indices = [t for t in [np.random.randint(x - x, m) for x in range(100)]]  # 生成100个0-m的随机数
    # display_data(x_train[rand_indices, :])  # 显示100个数字

    Lambda = 1  # 正则化系数
    all_theta = oneVsAll(x_train, y_train, num_labels, Lambda)  # 计算所有的theta

    predict_test = predict_oneVsAll(all_theta, x_test)  # 预测
    # 将预测结果和真实结果保存到文件中
    # np.savetxt("predict.csv", res, delimiter=',')
    predict_trian = predict_oneVsAll(all_theta, x_train)
    print("训练集预测:")
    show_accuracy(predict_trian, y_train)
    print("测试集预测:")
    show_accuracy(predict_test, y_test)

    # 显示出错的图
    err_num = np.sum(np.float64(predict_test != y_test).ravel())
    print('总共错误:', err_num)
    plot_error_image(x_test, y_test, predict_test, image_num=err_num, fig_image_num=[5, 5], image_size=[20, 20])


# 加载mat文件
def loadmat_data(fileName):
    return spio.loadmat(fileName)


# 显示100个数字
def display_data(imgData, image_num=[10, 10], image_size=[20, 20], pad=1):
    sum = 0
    '''
    显示100个数（若是一个一个绘制将会非常慢，可以将要画的数字整理好，放到一个矩阵中，显示这个矩阵即可）
    - 初始化一个二维数组
    - 将每行的数据调整成图像的矩阵，放进二维数组
    - 显示即可
    '''
    display_array = -np.ones((pad + image_num[0] * (image_size[0] + pad), pad + image_num[1] * (image_size[1] + pad)))
    # 十行十列
    for i in range(image_num[0]):
        for j in range(image_num[1]):
            display_array[pad + i * (image_size[0] + pad):pad + i * (image_size[0] + pad) + image_size[0],
            pad + j * (image_size[1] + pad):pad + j * (image_size[1] + pad) + image_size[1]] = (
                imgData[sum, :].reshape(image_size[0], image_size[1],
                                        order="F"))  # order=F指定以列优先，在matlab中是这样的，python中需要指定，默认以行
            sum += 1

    plt.imshow(display_array, cmap='gray')  # 显示灰度图像
    plt.axis('off')
    plt.show()


# 求每个分类的theta，最后返回所有的all_theta
def oneVsAll(X, y, num_labels, Lambda):
    # 初始化变量
    m, n = X.shape
    all_theta = np.zeros((num_labels, n + 1))  # 每一列对应相应分类的theta,共10行，每行对应0-9的训练结果
    X = np.hstack((np.ones((m, 1)), X))  # X前补上一列1的偏置bias
    class_y = np.zeros((m, num_labels))  # 数据的y对应0-9，需要映射为0/1的关系
    initial_theta = np.zeros((n + 1, 1))  # 初始化一个分类的theta

    # 映射y
    for i in range(num_labels):
        class_y[:, i] = np.int32(y == i).reshape(1, -1)  # 注意reshape(1,-1)才可以赋值

    # np.savetxt("class_y.csv", class_y[0:600,:], delimiter=',')

    '''遍历每个分类，计算对应的theta值'''
    for i in tqdm(range(num_labels)):
        # optimize.fmin_cg
        # result = computeGradientDescent(X, class_y[:, i],
        #                                 initial_theta, alpha = 0.1, inital_lambda = Lambda, epc = 1e-6)
        result = optimize.fmin_bfgs(costFunction, initial_theta, fprime=gradient,
                                    args=(X, class_y[:, i], Lambda))  # 调用梯度下降的优化方法
        all_theta[i, :] = result.ravel()  # 放入all_theta中
    # all_theta = np.transpose(all_theta)
    # np.savetxt("theta.csv", all_theta, delimiter=',')
    return all_theta


# 代价函数
def costFunction(initial_theta, X, y, inital_lambda):
    m = len(y)
    J = 0

    h = sigmoid(np.dot(X, initial_theta))  # 计算h(z)
    theta1 = initial_theta.copy()  # 因为正则化j=1从1开始，不包含0，所以复制一份，前theta(0)值为0
    theta1[0] = 0

    temp = np.dot(np.transpose(theta1), theta1)
    J = (-np.dot(np.transpose(y), np.log(h)) - np.dot(np.transpose(1 - y),
                                                      np.log(1 - h)) + temp * inital_lambda / 2) / m  # 正则化的代价方程
    return J


# 计算梯度
def gradient(initial_theta, X, y, inital_lambda):
    m = len(y)
    grad = np.zeros((initial_theta.shape[0]))

    h = sigmoid(np.dot(X, initial_theta))  # 计算h(z)
    theta1 = initial_theta.copy()
    theta1[0] = 0

    grad = np.dot(np.transpose(X), h - y) / m + inital_lambda / m * theta1  # 正则化的梯度
    return grad


def computeGradientDescent(X, y, theta, alpha, inital_lambda, epc):
    m = len(y)
    n = len(theta)
    y = y.reshape(-1, 1)
    # temp = np.matrix(np.zeros((n, num_iters)))  # 暂存每次迭代计算的theta，转化为矩阵形式

    # J_history = np.zeros((num_iters, 1))  # 记录每次迭代计算的代价值
    while (1):
        J0 = costFunction(theta, X, y, inital_lambda)
        h = sigmoid(np.dot(X, theta))  # 计算内积，matrix可以直接乘       h:118*1
        temp = theta - ((alpha / m) * (np.dot(np.transpose(X), h - y)))  # +inital_lambda/m*theta # 梯度的计算
        theta = temp
        J1 = costFunction(theta, X, y, inital_lambda)
        if (abs(J1 - J0) <= epc):  # 变化足够小时退出
            break

    return theta


# S型函数
def sigmoid(z):
    h = np.zeros((len(z), 1))  # 初始化，与z的长度一致

    h = 1.0 / (1.0 + np.exp(-z))
    return h


# 预测
def predict_oneVsAll(all_theta, X):
    m = X.shape[0]
    num_labels = all_theta.shape[0]

    X = np.hstack((np.ones((m, 1)), X))  # 在X最前面加一列1

    h = sigmoid(np.dot(X, np.transpose(all_theta)))  # 预测
    '''
    返回h中每一行最大值所在的列号
    - np.max(h, axis=1)返回h中每一行的最大值（是某个数字的最大概率）
    - 最后where找到的最大概率所在的列号（列号即是对应的数字）
    '''
    p = np.zeros((m, 1))
    for i in range(m):
        t = np.array(np.where(h[i, :] == np.max(h, axis=1)[i]))
        p[i] = t
    return p


# 显示准确度
def show_accuracy(p, y):
    print(u"正确率为：%f%%" % np.mean(np.float64(p == y) * 100))


# 显示分类出错的图片
def plot_error_image(x_test, y_test, predict_test, image_num, fig_image_num, image_size=[20, 20]):

    """
    image_num:      总共图片数
    fig_image_num：  每个画布内图片行列数
    image_size：     每张图片大小
    """
    err_images = x_test[np.array(np.where(y_test != predict_test))[0]]
    err_images = err_images.reshape(-1, image_size[0], image_size[1])
    err_y_hat = predict_test[np.array(np.where(y_test != predict_test))[0]].ravel()
    err_y = y_test[np.array(np.where(y_test != predict_test))[0]].ravel()
    #     plt.figure(figsize=(10, 8), facecolor='w')
    i = 0
    for index, image in enumerate(err_images):
        if index >= image_num:
            break
        if i > (fig_image_num[0] * fig_image_num[1] - 1):
            i = 0
            plt.figure()
        plt.subplot(fig_image_num[0], fig_image_num[1], i + 1)
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title(u'错分为：%i，真实值：%i' % (err_y_hat[index], err_y[index]))
        i += 1
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    logisticRegression_OneVsAll()
