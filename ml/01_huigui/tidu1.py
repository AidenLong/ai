# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False


def f(x):
    return x ** 2


def h(x):
    return 2 * x


X = []
Y = []
x = 2
step = 0.8
f_change = f(x)
f_current = f(x)
X.append(x)
Y.append(f_current)
while f_change > 1e-10:
    x = x - step * h(x)
    tmp = f(x)
    f_change = np.abs(f_current - tmp)
    f_current = tmp
    X.append(x)
    Y.append(f_current)
    print(f_change)

print('最终结果为', (x, f_current))

fig = plt.figure()
X2 = np.arange(-2, 2.2, 0.05)
Y2 = X2 ** 2
plt.plot(X2, Y2, '-', color='#666666', linewidth=2)
plt.plot(X, Y, 'bo--')
plt.title('$y=x**2函数求解最小值，最终解为:x=%.2f,y=%.2f' % (x, f_current))
plt.show()
