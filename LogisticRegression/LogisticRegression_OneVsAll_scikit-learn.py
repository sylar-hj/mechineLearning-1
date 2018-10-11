# -*- coding: utf-8 -*-
from scipy import io as spio
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def logisticRegression_oneVsAll():
    data = loadmat_data("data_digits.mat")
    X = data['X']  # 获取X数据，每一行对应一个数字20x20px
    y = data['y']  # 这里读取mat文件y的shape=(5000, 1)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    y_test = np.ravel(y_test)  # 调用sklearn需要转化成一维的(5000,)
    y_train = np.ravel(y_train)

    model = LogisticRegression()
    model.fit(x_train, y_train)  # 拟合

    predict = model.predict(x_test)  # 预测

    print(u"测试集预测准确度为：%f%%" % np.mean(np.float64(predict == y_test) * 100))


# 加载mat文件
def loadmat_data(fileName):
    return spio.loadmat(fileName)


if __name__ == "__main__":
    logisticRegression_oneVsAll()
