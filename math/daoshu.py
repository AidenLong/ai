# -*- coding:utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt


# 函数单调性
def f(x):
    return 3 * x * x + 5 * x + 8


def h(x):
    return 6 * x + 5


x = np.arange(-4, 2.4, 0.05)
y1 = [f(i) for i in x]
y2 = [h(i) for i in x]
plt.plot(x, y1, linewidth=2, color='y', label='3 * x * x + 5 * x + 8')
plt.plot(x, y2, linewidth=2, color='b', label='6 * x + 5')
plt.plot([-5.0 / 6, -5.0 / 6], [-20, 40], '--', color="#999999", linewidth=2)
plt.legend(loc='lower right')
plt.grid(True)
plt.show()