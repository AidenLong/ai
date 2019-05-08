# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd

# 输入系数矩阵（X和Y）
X = pd.DataFrame([[2.0, -1.0], [3.0, 1.0]])
Y = pd.Series([2.0, 1.0])
# 特征列单位化（长度是1）
a = lambda x: x / np.sqrt(np.dot(x, x))
X[0] = a(X.iloc[:, 0])
X[1] = a(X.iloc[:, 1])
print(X)

# 计算相关系数
w = lambda x, Y: np.dot(x, Y)

# 初始残差
Yres = Y
XX = pd.DataFrame()
# 选择相关系数最大的特征列
if w(X[0], Yres) > w(X[1], Yres):
    w0 = 0
else:
    w0 = 1
# 获取最大相关系数
W0 = w(X[w0], Yres)
# 第一个特征
XX = XX.append(X[w0])
print(X[w0])
# 新的残差
Yres = Yres - W0 * X[w0]
# 剩余的特征
print(Yres)

XX = XX.append(X[1])
# 第二轮
W1 = w(X[1], Yres)
# 最终残差
Yres = Yres - W1 * X[1]
Y_ = np.sqrt(np.dot(Yres, Yres))
# 系数
W = [W0, W1]

print("最终的残差：", Yres, "误差距离：", Y_)
print("系数W:", W, "\n", "对应的X：", XX)

print(W0 / np.sqrt(4 + 9))
print(W1 / np.sqrt(1 + 1))
