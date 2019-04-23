# -*- coding:utf-8 -*-
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor


def notEmpty(s):
    return s != ''


mpl.rcParams['font.sans-serif'] = ['simHei']
mpl.rcParams['axes.unicode_minus'] = False

names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
path = "datas/boston_housing.data"
#
fd = pd.read_csv(path, header=None)
data = np.empty((len(fd), 14))
for i, d in enumerate(fd.values):
    d = map(float, filter(notEmpty, d[0].split(' ')))
    data[i] = list(d)

x, y = np.split(data, (13,), axis=1)
y = y.ravel()

print('样本数据量：%d，特征个数：%d' % x.shape)
print('target样本数量：%d' % y.shape[0])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=14)
print("训练数据集样本数目：%d, 测试数据集样本数目：%d" % (x_train.shape[0], x_test.shape[0]))

# 构建线性回归
lr = LinearRegression()
lr.fit(x_train, y_train)
lr_y_test_hat = lr.predict(x_test)
lr_score = lr.score(x_test, y_test)
print('lr:', lr_score)

# 构建一个Bagging的线性模型
bg = BaggingRegressor(LinearRegression(), n_estimators=100, max_samples=0.7, random_state=28)
bg.fit(x_train, y_train)
bg_y_test_hat = bg.predict(x_test)
bg_score = bg.score(x_test, y_test)
print('Bagging:', bg_score)

# 构建一个AdaBoost算法的线性模型
from sklearn.ensemble import AdaBoostRegressor

# 在当前情况下，单个的LinearRegression回归的模型的效果还不错(60%+)
abr = AdaBoostRegressor(LinearRegression(), n_estimators=50, learning_rate
=0.0001, random_state=28)
abr.fit(x_train, y_train)
abr_y_test_hat = abr.predict(x_test)
abr_score = abr.score(x_test, y_test)
print("AadaBoost:", abr_score)

# GBDT
from sklearn.ensemble import GradientBoostingRegressor

gbdt = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, random_state=28)
gbdt.fit(x_train, y_train)
gbdt_y_test_hat = gbdt.predict(x_test)
gbdt_score = gbdt.score(x_test, y_test)
print("GBDT:", gbdt_score)
