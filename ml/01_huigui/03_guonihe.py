# -*- coding:utf-8 -*-
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model.coordinate_descent import ConvergenceWarning

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import sklearn

## 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False
## 拦截异常
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)

## 创建模拟数据
np.random.seed(100)
np.set_printoptions(linewidth=1000, suppress=True)  # 显示方式设置，每行的字符数用于插入换行符，是否使用科学计数法
N = 10
x = np.linspace(0, 6, N) + np.random.randn(N)
y = 1.8 * x ** 3 + x ** 2 - 14 * x - 7 + np.random.randn(N)

## 将其设置为矩阵
x.shape = -1, 1
y.shape = -1, 1

# RidgeCV和Ridge的区别是，前者可以进行交叉验证
models = [
    Pipeline([
        ('Poly', PolynomialFeatures(include_bias=False)),
        ('Linear', LinearRegression(fit_intercept=False))
    ]),
    Pipeline([
        ('Poly', PolynomialFeatures(include_bias=False)),
        ('Linear', RidgeCV(alphas=np.logspace(-3, 2, 50), fit_intercept=False))
    ]),
    Pipeline([
        ('Poly', PolynomialFeatures(include_bias=False)),
        ('Linear', LassoCV(alphas=np.logspace(0, 1, 10), fit_intercept=False))
    ]),
    Pipeline([
        ('Poly', PolynomialFeatures(include_bias=False)),
        ('Linear', ElasticNetCV(alphas=np.logspace(0, 1, 10), l1_ratio=[.1, .5, .7, .9, .95, 1], fit_intercept=False))
    ]),
]

# 线性模型过拟合图形识别
plt.figure(facecolor='w')
degree = np.arange(1, N, 4)
dm = degree.size
colors = []  # 颜色
for c in np.linspace(16711680, 255, dm):
    colors.append('#%06x' % int(c))

model = models[0]
for i, d in enumerate(degree):
    plt.subplot(int(np.ceil(dm / 2.0)), 2, i + 1)  # np.ceil 计算大于等于该值的最小整数
    plt.plot(x, y, 'ro', ms=10, zorder=N)

    # 设置阶数
    model.set_params(Poly__degree=d)
    # 模型训练
    model.fit(x, y.ravel())  # ravel多维数组转以为数组

    lin = model.get_params('Linear')['Linear']
    output = u'%d阶，系数为：' % d
    print(output, lin.coef_.ravel())

    x_hat = np.linspace(x.min(), x.max(), num=100)
    x_hat.shape = -1, 1
    y_hat = model.predict(x_hat)
    s = model.score(x, y)

    z = N - 1 if (d == 2) else 0
    label = u'%d阶，正确率=%.3f' % (d, s)
    plt.plot(x_hat, y_hat, color=colors[i], lw=2, alpha=0.75, label=label, zorder=z)

    plt.legend(loc='upper left')
    plt.grid(True)
    plt.xlabel('X', fontsize=16)
    plt.ylabel('Y', fontsize=16)

plt.tight_layout(1, rect=(0, 0, 1, 0.95))
plt.suptitle(u'线性回归过拟合显示', fontsize=22)
plt.show()


## 线性回归、Lasso回归、Ridge回归、ElasticNet比较
plt.figure(facecolor='w')
degree = np.arange(1, N, 2)  # 阶
dm = degree.size
colors = []  # 颜色
for c in np.linspace(16711680, 255, dm):
    colors.append('#%06x' % int(c))
titles = [u'线性回归', u'Ridge回归', u'Lasso回归', u'ElasticNet']

for t in range(4):
    model = models[t]  # 选择了模型--具体的pipeline
    plt.subplot(2, 2, t + 1)
    plt.plot(x, y, 'ro', ms=10, zorder=N)

    for i, d in enumerate(degree):
        # 设置阶数(多项式)
        model.set_params(Poly__degree=d)
        # 模型训练
        model.fit(x, y.ravel())

        # 获取得到具体的算法模型
        lin = model.get_params('Linear')['Linear']
        # 打印数据
        output = u'%s:%d阶，系数为：' % (titles[t], d)
        print(output, lin.coef_.ravel())

        # 产生模拟数据
        x_hat = np.linspace(x.min(), x.max(), num=100)  ## 产生模拟数据
        x_hat.shape = -1, 1
        # 数据预测
        y_hat = model.predict(x_hat)
        # 计算准确率
        s = model.score(x, y)

        #
        z = N - 1 if (d == 2) else 0
        label = u'%d阶, 正确率=%.3f' % (d, s)
        plt.plot(x_hat, y_hat, color=colors[i], lw=2, alpha=0.75, label=label, zorder=z)

    plt.legend(loc='upper left')
    plt.grid(True)
    plt.title(titles[t])
    plt.xlabel('X', fontsize=16)
    plt.ylabel('Y', fontsize=16)
plt.tight_layout(1, rect=(0, 0, 1, 0.95))
plt.suptitle(u'各种不同线性回归过拟合显示', fontsize=22)
plt.show()
