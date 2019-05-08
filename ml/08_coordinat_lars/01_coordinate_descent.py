# -*- coding:utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# 解决中文显示问题
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 输入系数矩阵（X和Y）
X = pd.DataFrame([[2.0, -1.0], [3.0, 1.0]])
Y = pd.Series([2.0, 1.0])


# 目标函数
def f(X, w, Y):
    y1 = pd.Series([np.dot(X.iloc[0, :], w), np.dot(X.iloc[1, :], w)])
    return 1 / 2 * np.dot(Y - y1, Y - y1)


##迭代函数
def h0(w, X, Y):
    y1 = pd.Series([np.dot(X.iloc[0, :], w), np.dot(X.iloc[1, :], w)])
    return w[0] + np.dot(X.iloc[:, 0], (Y - y1)) / np.dot(X.iloc[:, 0], X.iloc[:, 0])


def h1(w, X, Y):
    y1 = pd.Series([np.dot(X.iloc[0, :], w), np.dot(X.iloc[1, :], w)])
    return w[1] + np.dot(X.iloc[:, -1], (Y - y1)) / np.dot(X.iloc[:, -1], X.iloc[:, -1])


# 初始化
XX = []
YY = []
Z = []
w = pd.Series([0.0, 0.0])
y1 = pd.Series([np.dot(X.iloc[0, :], w), np.dot(X.iloc[1, :], w)])
f_change = 1 / 2 * np.dot(Y - y1, Y - y1)
f_current = f(X, w, Y)
# 迭代流程
while f_change > 1e-10:
    w[0] = h0(w, X, Y)
    w[1] = h1(w, X, Y)
    f_change = f_current - f(X, w, Y)
    f_current = f(X, w, Y)
    XX.append(w[0])
    YY.append(w[1])
    Z.append(f_current)
print(u"最终结果为:", w)
print(XX, "\n", YY, "\n", Z)

fig = plt.figure()
ax = Axes3D(fig)
X2 = np.arange(-1, 1, 0.05)
Y2 = np.arange(-1, 1, 0.05)
X2, Y2 = np.meshgrid(X2, Y2)
Z2 = 1 / 2 * (2 - 2 * X2 + Y2) ** 2 + 1 / 2 * (1 - 3 * X2 - Y2) ** 2
ax.plot_surface(X2, Y2, Z2, cmap='rainbow')
ax.plot(XX, YY, Z, 'ro--')
ax.set_title(u'坐标下降求解, 最终解为: w0=%.2f, w1=%.2f, z=%.2f' % (w[0], w[1], f_current))

plt.show()
