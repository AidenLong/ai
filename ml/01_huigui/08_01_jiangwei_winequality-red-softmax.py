# -*- coding:utf-8 -*-
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from sklearn.exceptions import DataConversionWarning
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model.coordinate_descent import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import Normalizer

## 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False
## 拦截异常
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=Warning)

## 读取数据
path1 = "datas/winequality-red.csv"
df1 = pd.read_csv(path1, sep=";")
df1['type'] = 1  # 设置数据类型为红葡萄酒

path2 = "datas/winequality-white.csv"
df2 = pd.read_csv(path2, sep=";")
df2['type'] = 2  # 设置数据类型为白葡萄酒

# 合并两个df
df = pd.concat([df1, df2], axis=0)

## 自变量名称
names = ["fixed acidity", "volatile acidity", "citric acid",
         "residual sugar", "chlorides", "free sulfur dioxide",
         "total sulfur dioxide", "density", "pH", "sulphates",
         "alcohol", "type"]
## 因变量名称
quality = "quality"

## 显示
print(df.head(5))

## 异常数据处理
new_df = df.replace('?', np.nan)
datas = new_df.dropna(how='any')  # 只要有列为空，就进行删除操作
print("原始数据条数:%d；异常数据处理后数据条数:%d；异常数据条数:%d" % (len(df), len(datas), len(df) - len(datas)))

## 提取自变量和因变量
X = datas[names]
Y = datas[quality]

## 数据分割
X1_train, X1_test, Y1_train, Y1_test = train_test_split(X, Y, test_size=0.25, random_state=0)
print("训练数据条数:%d；数据特征个数:%d；测试数据条数:%d" % (X1_train.shape[0], X1_train.shape[1], X1_test.shape[0]))

# 2.数据归一化（归一化）
# ss2 = Normalizer()
# X1_train = ss2.fit_transform(X1_train)

## 特征选择
skb = SelectKBest(chi2, k=3)  ## 只考虑3个维度
X1_train = skb.fit_transform(X1_train, Y1_train)  ## 训练模型及特征选择

## 降维
# pca = PCA(n_components=5) ## 将样本数据维度降低成为2个维度
# X1_train = pca.fit_transform(X1_train)
# print("贡献率:", pca.explained_variance_)

## 模型构建
lr2 = LogisticRegressionCV(fit_intercept=True, Cs=np.logspace(-5, 1, 100),
                           multi_class='multinomial', penalty='l2', solver='lbfgs')
lr2.fit(X1_train, Y1_train)

## 模型效果输出
r = lr2.score(X1_train, Y1_train)
print("R值：", r)
print("特征稀疏化比率：%.2f%%" % (np.mean(lr2.coef_.ravel() == 0) * 100))
print("参数：", lr2.coef_)
print("截距：", lr2.intercept_)

## 数据预测
## a. 预测数据格式化(归一化)
# X1_test = ss2.transform(X1_test)  ## 测试数据归一化
X1_test = skb.transform(X1_test) ## 测试数据特征选择
# X1_test = pca.fit_transform(X1_test) ## 测试数据降维

## b. 结果数据预测
Y1_predict = lr2.predict(X1_test)

## 图表展示
## c. 图表展示
x1_len = range(len(X1_test))
plt.figure(figsize=(14, 7), facecolor='w')
plt.ylim(-1, 11)
plt.plot(x1_len, Y1_test, 'ro', markersize=8, zorder=3, label=u'真实值')
plt.plot(x1_len, Y1_predict, 'go', markersize=12, zorder=2, label=u'预测值,$R^2$=%.3f' % lr2.score(X1_train, Y1_train))
plt.legend(loc='upper left')
plt.xlabel(u'数据编号', fontsize=18)
plt.ylabel(u'葡萄酒质量', fontsize=18)
plt.title(u'葡萄酒质量预测统计(降维处理)', fontsize=20)
plt.show()
