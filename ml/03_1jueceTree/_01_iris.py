# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# 定义一些常量
iris_feature_E = 'sepal length', 'sepal width', 'petal length', 'petal width'
iris_feature_C = '花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度'
iris_class = 'Iris-setosa', 'Iris-versicolor', 'Iris-virginica'

# 1. 加载数据
path = './datas/iris.data'
names = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
df = pd.read_csv(path, sep=',', header=None, names=names)

# 2. 查看数据的相关特性
print(df.head(5))
print(df.info())
print(df.describe())

# 3. 从原始数据中获取X和Y
X = df[names[:-1]]  # 获取前4列的数据, 根据列名称获取
Y = df[names[-1]]   # 获取最后一列的数据，根据列名称获取
Y = pd.Categorical(Y).codes # 将类别转换为数字

# 4. 数据划分
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.7, random_state=28)
print('训练集样本数量：%d' % x_train.shape[0])
print('测试集样本数量：%d' % x_test.shape[0])

# 5. 模型训练 ==》 使用决策树相关模型进行分类
# scikit-learn中的决策树默认都是CART模型（使用gini系数作为数据纯度的量化指标）
# criterion='gini' ==> 给定书构建过程中需要考虑的指标，默认为gini系数也就是CART，可选entropy，表示使用信息熵
# splitter='best' ==> 给定选择划分属性的时候，采用什么方式，默认为best，表示选择最优的方式划分，可选random，表示随机
# max_depth=None ==> 树的最大允许深度，默认不限制
# min_samples_split=2 ==> 当节点中样本数量小于等于该值的时候，停止树的构建
tree = DecisionTreeClassifier(criterion='entropy', max_depth=2)
tree.fit(x_train, y_train)

# 模型相关的指标输出
print('训练集上的准确率:%.3f' % tree.score(x_train, y_train))
print('测试集上的准确率:%.3f' % tree.score(x_test, y_test))

# 6. 模型的预测
print('直接预测所属类别')
print(tree.predict(x_test))
print('每个样本的预测概率信息:')
print(tree.predict_proba(x_test))

# 在决策树构建的时候，将影响信息增益更大的属性是放在树的上面的节点进行判断的，也就是说
# 可以认为决策树构建的树中，越往上的节点，作用越强==> 所以可以基于决策树做特征选择
# 实际代码中，其实就是feature_importance 参数输出值中最大的k个指标
print('各个特征中的重要指标：', end='')
print(tree.feature_importances_)

# 支持通过joblib来保存和加载模型
from sklearn.externals import joblib
joblib.dump(tree, 'tree.model')
tree2 = joblib.load('tree.model')
print(tree2.predict(x_test))
