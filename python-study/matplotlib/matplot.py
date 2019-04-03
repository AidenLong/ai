# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# 解决中文显示问题
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 获取figure对象
fig = plt.figure(figsize=(8,6))
# 在figure上创建axes对象
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
# 在当前axes上绘制曲线（ax3）
plt.plot(np.random.randn(50).cumsum(), 'k--')
# 在ax1上绘制柱状图
ax1.hist(np.random.randn(300), bins=20, color='k', alpha=0.3)
# 在ax2上绘制散点图
ax2.scatter(np.arange(30), np.arange(30) + 3 * np.random.randn(30))

# 展示
plt.show()


