# -*- coding:utf-8 -*-
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# 产生一个随机数
rng = np.random.RandomState(1)
x = np.sort(10 * rng.rand(80, 1), axis=0)
y = np.sin(x).ravel()

y[::5] += 3 * (0.5 - rng.rand(16))
# 构建不同深度的决策树
clf_0 = DecisionTreeRegressor(max_depth=1)
clf_1 = DecisionTreeRegressor(max_depth=2)
clf_2 = DecisionTreeRegressor(max_depth=3)
clf_3 = DecisionTreeRegressor(max_depth=4)
clf_0.fit(x, y)
clf_1.fit(x, y)
clf_2.fit(x, y)
clf_3.fit(x, y)
# 创建预测模拟数据
x_test = np.arange(0.0, 10, 0.01)[:, np.newaxis]
y_0 = clf_0.predict(x_test)
y_1 = clf_1.predict(x_test)
y_2 = clf_2.predict(x_test)
y_3 = clf_3.predict(x_test)

# 图表展示
plt.figure(figsize=(16, 9), dpi=80, facecolor='w')
plt.scatter(x, y, c="k", s=10, label="data")
plt.plot(x_test, y_0, c="y", label="max_depth=1,$R^2$=%.3f" % (clf_0.score(x, y)), linewidth=2)
plt.plot(x_test, y_1, c="g", label="max_depth=2,$R^2$=%.3f" % (clf_1.score(x, y)), linewidth=2)
plt.plot(x_test, y_2, c="r", label="max_depth=3,$R^2$=%.3f" % (clf_2.score(x, y)), linewidth=2)
plt.plot(x_test, y_3, c="b", label="max_depth=5,$R^2$=%.3f" % (clf_3.score(x, y)), linewidth=2)
plt.xlabel("X", horizontalalignment="left")
plt.ylabel("Y")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()
