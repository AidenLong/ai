# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False


# 数据加载
path = "datas/iris.data"
names = ['sepal length', 'sepal width', 'petal length', 'petal width', 'cla']
df = pd.read_csv(path, header=None, names=names)


def parseRecord(record):
    result = []
    r = zip(names, record)
    for name, v in r:
        if name == 'cla':
            if v == 'Iris-setosa':
                result.append(1)
            elif v == 'Iris-versicolor':
                result.append(2)
            elif v == 'Iris-virginica':
                result.append(3)
            else:
                result.append(np.nan)
        else:
            result.append(float(v))
    return result


# 1. 数据转换为数字以及分割
# 数据转换
datas = df.apply(lambda r: pd.Series(parseRecord(r), index=names), axis=1)
# 异常数据删除
datas = datas.dropna(how='any')
print(datas.head())
# 数据分割
X = datas[names[0:-1]]
Y = datas[names[-1]]
# 数据抽样(训练数据和测试数据分割)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
print("训练样本数量:%d,特征属性数目:%d" % (x_train.shape[0], x_train.shape[1]))
print("测试样本数量:%d" % x_test.shape[0])

model = XGBClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
predictions = [round(value) for value in y_pred]

# 评估预测结果
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

x_test_len = range(len(x_test))
plt.figure(figsize=(12, 9), facecolor='w')
plt.ylim(0.5, 3.5)
plt.plot(x_test_len, y_test, 'ro', markersize=6, zorder=3, label='真实值')
plt.plot(x_test_len, y_pred, 'yo', markersize=12, zorder=1,
         label='xgboost算法预测值,$R^2$=%.3f' % accuracy_score(y_test, predictions))
plt.legend(loc='lower right')
plt.xlabel('数据编号', fontsize=18)
plt.ylabel('种类', fontsize=18)
plt.title('莺尾花数据分类', fontsize=18)
plt.show()
