# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV

# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False


def notEmpty(s):
    return s != ''


# 读取数据
df = pd.read_csv('boston_housing.data', header=None)
# print(df.head())
data = np.empty((len(df), 14))
# 数据处理
for i, d in enumerate(df.values):
    d = map(float, filter(notEmpty, d[0].split(' ')))
    data[i] = list(d)
# print(data)

# 数据分割
x, y = np.split(data, (13,), axis=1)
# y进行格式转换，拉直
y = y.ravel()
# print(x[:3])
# print(y[:3])

# 数据分割
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# 使用Lasso进行特征选择
lso = Lasso()
lso.fit(x_train, y_train)
print(lso.coef_)
'''
[-0.05889028  0.05317657 -0.          0.         -0.          0.67954962
  0.01684077 -0.6487664   0.198738   -0.01399421 -0.86421958  0.00660309
 -0.73120957]
'''
# 删除coef_为0的特征
x_train = np.hstack((x_train[:, :2], x_train[:, 5:]))
x_test = np.hstack((x_test[:, :2], x_test[:, 5:]))

# Pieline 并行调参
models = [
    Pipeline([
        ('ss', StandardScaler()),
        ('poly', PolynomialFeatures()),
        ('linear', LinearRegression())
    ]),
    Pipeline([
        ('ss', StandardScaler()),
        ('poly', PolynomialFeatures()),
        ('linear', Lasso())
    ]),
    Pipeline([
        ('ss', StandardScaler()),
        ('poly', PolynomialFeatures()),
        ('linear', Ridge())
    ]),
    Pipeline([
        ('ss', StandardScaler()),
        ('poly', PolynomialFeatures()),
        ('linear', ElasticNet())
    ])
]

# 参数字典，字典中的key是属性的名字，value是可选的参数列表
parameters = {
    'poly__degree': [3, 2, 1]
}

titles = ['Linear', 'Lasso', 'Ridge', 'ElasticNet']
for t in range(4):
    # 获取模型并设置参数
    model = GridSearchCV(models[t], param_grid=parameters, cv=5, n_jobs=1)
    # 模型训练
    model.fit(x_train, y_train)
    # 模型效果值获取（最优参数）
    print("%s算法:最优参数:" % titles[t], model.best_params_)
    print("%s算法:R值=%.3f" % (titles[t], model.best_score_))
