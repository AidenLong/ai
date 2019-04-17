# -*- coding:utf-8 -*-

import numpy as np
import math

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib as mpl
import matplotlib.pyplot as plt

'''
    实现基于批量梯度下降的线性回归算法
'''
## 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False


def predict(x, theta, intercept=0.0):
    '''
    碎语样本x产生的预测值
    :param x:
    :param theta:
    :param intercept:
    :return:
    '''
    result = 0.0
    # 1.x和theta相乘
    n = len(x)
    for i in range(n):
        result += x[i] * theta[i]
    # 2.加上截距项
    result += intercept
    # 返回结果
    return result


def predict_X(X, theta, ntercept=0.0):
    Y = []
    for x in X:
        Y.append(predict(x, theta, intercept))
    return Y


def fit(X, Y, alpha=0.01, max_iter=None, tol=1e-5, fit_intercept=True):
    '''
    使用批量的梯度下降实现线性回归的参数求解，最终返回参数theta以及截距项intercept
    :param X:   输入的特征矩阵，格式是二维的矩阵形式m*n,m表示样本数目，n表示每个样本的特征属性数目
    :param Y:   输入的目标属性矩阵，格式是二维数组形式，m*1,m表示样本数，1表示预测属性y
    :param alpha:   学习率
    :param max_iter:    最大迭代次数
    :param tol: 收敛值
    :param fit_intercept:   是否训练截距项
    :return:
    '''

    # 1.预处理
    X = np.array(X)
    Y = np.array(Y)
    if max_iter is None:
        max_iter = 200
    max_iter = max_iter if max_iter > 0 else 200

    # 2.获取样本的数目和维度信息
    m, n = np.shape(X)

    # 3.定义相关的变量
    theta = np.zeros(shape=[n])
    intercept = 0
    # 定义一个存储所有样本的残差的数组
    diff = np.zeros(shape=[m])
    # 定义上一个迭代的随时函数的值
    pred_j = 1 << 64

    # 4.开始模型迭代
    for i in range(max_iter):
        # a.计算所有样本的预测值和实际值之间的差值
        for k in range(m):
            y_true = Y[k][0]
            y_predict = predict(X[k], theta, intercept)
            diff[k] = y_true - y_predict

        # b。基于所有样本的预测值和实际值之间的差值更新每一个theta值
        for j in range(n):
            # --1.累加所有样本的梯度值
            gd = 0
            for k in range(m):
                gd += diff[k] * X[k][j]
            # --2.基于梯度值更新模型参数
            theta[j] += alpha * gd

        # c。基于所有样本的预测值和实际值之间的差值更新截距项
        if fit_intercept:
            # --1.累加所有的梯度值
            gd = np.sum(diff)
            # --2.参数更新
            intercept += alpha * gd

        # d。计算在当前模型参数的情况下，损失函数的值
        sum_j = 0.0
        for k in range(m):
            y_true = Y[k][0]
            y_predict = predict(X[k], theta, intercept)
            sum_j += math.pow(y_true - y_predict, 2)
        sum_j /= 2.0

        # e。比较上一次的损失函数值和当前损失函数值之间的差值是否小于参数tol，如果小于则结束
        if np.abs(pred_j - sum_j) < tol:
            break
        pred_j = sum_j
    print('迭代{}次后，损失函数值为:{}'.format(i, pred_j))
    return theta, intercept, pred_j


if __name__ == '__main__':
    # 1.构建模拟数据
    np.random.seed(28)
    N = 10
    x = np.linspace(start=0, stop=6, num=N) + np.random.rand(N)
    y = 1.8 * x ** 3 + x ** 2 - 14 * x - 7 + np.random.rand(N)
    x.shape = -1, 1
    y.shape = -1, 1

    # 2.使用算法构建模型
    model = LinearRegression()
    model.fit(x, y)
    s1 = model.score(x, y)
    print("sklearn内置线性回归模型==========")
    print('训练数据R2：{}'.format(s1))
    print('模型的参数值：{}'.format(model.coef_))
    print('模型的截距项：{}'.format(model.intercept_))
    theta, intercept, sum_j = fit(x, y)
    s2 = r2_score(y, predict_X(x, theta, intercept))
    print("自定义基于梯度下降线性回归模型==========")
    print('训练数据R2：{}'.format(s2))
    print('模型的参数值：{}'.format(theta))
    print('模型的截距项：{}'.format(intercept))

    ## 模拟数据产生
    x_hat = np.linspace(x.min(), x.max(), num=100)
    x_hat.shape = -1, 1
    y_hat = model.predict(x_hat)
    y_hat2 = predict_X(x_hat, theta, intercept)

    ## 开始画图
    plt.plot(x, y, 'ro', ms=10, zorder=3)
    plt.plot(x_hat, y_hat, color='#b624db', lw=2, alpha=0.75, label=u'sklearn内置线性回归模型，$R^2$:%.3f' % s1, zorder=2)
    plt.plot(x_hat, y_hat2, color='#6d49b6', lw=2, alpha=0.75, label=u'自己实现模型，$R^2$:%.3f' % s2, zorder=1)
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.xlabel('X', fontsize=16)
    plt.ylabel('Y', fontsize=16)

    plt.suptitle(u'自定义的线性模型和模块中的线性模型比较', fontsize=22)
    plt.show()
