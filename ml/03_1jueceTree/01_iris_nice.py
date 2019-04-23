# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
from sklearn import tree  # 决策树
from sklearn.tree import DecisionTreeClassifier  # 分类树
from sklearn.model_selection import train_test_split  # 测试集和训练集
from sklearn.pipeline import Pipeline  # 管道
from sklearn.feature_selection import SelectKBest  # 特征选择
from sklearn.feature_selection import chi2  # 卡方统计量

from sklearn.preprocessing import MinMaxScaler  # 数据归一化
from sklearn.decomposition import PCA  # 主成分分析
from sklearn.model_selection import GridSearchCV  # 网格搜索交叉验证，用于选择最优的参数

## 设置属性防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 读取数据
path = './datas/iris.data'
data = pd.read_csv(path, header=None)
x = data[list(range(4))]  # 获取X变量
y = pd.Categorical(data[4]).codes  # 把Y转换成分类型的0,1,2
print("总样本数目：%d;特征属性数目:%d" % x.shape)
print(data.head(5))

# 数据进行分割（训练数据和测试数据）
x_train1, x_test1, y_train1, y_test1 = train_test_split(x, y, train_size=0.8, random_state=14)
x_train, x_test, y_train, y_test = x_train1, x_test1, y_train1, y_test1
print("训练数据集样本数目：%d, 测试数据集样本数目：%d" % (x_train.shape[0], x_test.shape[0]))
y_train = y_train.astype(np.int)
y_test = y_test.astype(np.int)

# 数据标准化
# StandardScaler (基于特征矩阵的列，将属性值转换至服从正态分布)
# 标准化是依照特征矩阵的列处理数据，其通过求z-score的方法，将样本的特征值转换到同一量纲下
# 常用语基于正态分布的算法，比如回归

# 数据归一化
# MinMaxScaler (区间缩放，基于最大最小值，将数据转换为0,1区间上的)
# 提升模型收敛速度，提升模型精度
# 常用于神经网络

# Normalizer（基于矩阵的列，将样本向量转换为单位向量）
# 其目的在于样本向量在点成运算或其他核函数计算相似性时，拥有统一的标准
# 常见用于文本分类和聚类，logistic回归中也会使用，有效防止过拟合

ss = MinMaxScaler()
# 用标准化方法对数据进行处理并转换
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)
print("原始数据各个特征属性的调整最小值:", ss.min_)
print("原始数据各个特征属性的缩放数据值:", ss.scale_)

# 特征选择：从已有的特征中选择出影响目标值最大的特征属性
# 常用方法：
#   （分类：F统计量、卡方系数，互信息mutual_info_classif
#       连续：皮尔逊相关系数，F统计量、互信息mutual_info_classif）
# SelectKBest（卡方系数）

ch2 = SelectKBest(chi2, k=3)  # 在当前的案例中，使用SelectKBest这个方法从4个原始的特征属性，选择出来3个
# K 默认为10，如果指定了，那么会返回你说想要的特征的个数
x_train = ch2.fit_transform(x_train, y_train)
x_test = ch2.transform(x_test)

select_name_index = ch2.get_support(indices=True)
print("对类别判断影响最大的三个特征属性分布是:", ch2.get_support(indices=False))
print(select_name_index)

# 降维：对于数据而言，如果特征属性比较多，在构建过程中，会比较复杂，这个时候考虑将多维（高维）映射到低维的数据
# 常用的方法
# PCA：主成分分析（无监督）
# LDA：线性判别分析（有监督）类内方差最小，人脸识别，通常先做一次pca

pca = PCA(n_components=2)  # 构建一个pca对象，设置最终维度是2维
# #这里是为了后面画图方便，所以将数据维度设置了2维，一般用默认不设置参数就可以

x_train = pca.fit_transform(x_train, y_train)
x_test = pca.transform(x_test)

# 模型构建
model = DecisionTreeClassifier(criterion='entropy', random_state=0)
# 模型训练
model.fit(x_train, y_train)
# 模型预测
y_test_hat = model.predict(x_test)

# 模型结果的评估
y_test2 = y_test.reshape(-1)
result = (y_test2 == y_test_hat)
print('准确率:%.2f' % (np.mean(result) * 100))
# 实际可通过参数获取
print('Score:', model.score(x_test, y_test))
print('Classes:', model.classes_)

# 画图
N = 100  # 横纵和采样多少个值
x1_min = np.min((x_train.T[0].min(), x_test.T[0].min()))
x1_max = np.min((x_train.T[0].max(), x_test.T[0].max()))
x2_min = np.min((x_train.T[1].min(), x_test.T[1].min()))
x2_max = np.min((x_train.T[1].max(), x_test.T[1].max()))

t1 = np.linspace(x1_min, x1_max, N)
t2 = np.linspace(x2_min, x2_max, N)
x1, x2 = np.meshgrid(t1, t2)  # 生成网络采样点
x_show = np.dstack((x1.flat, x2.flat))[0]  # 测试点

y_show_hat = model.predict(x_show)  # 预测值

y_show_hat = y_show_hat.reshape(x1.shape)  # 使之与输入的形状相同
print(y_show_hat.shape)
print(y_show_hat[0])

# 画图
plt_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
plt_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])

plt.figure(facecolor='w')
plt.pcolormesh(x1, x2, y_show_hat, cmap=plt_light)
plt.scatter(x_test.T[0], x_test.T[1], c=y_test.ravel(), edgecolors='k', s=150, zorder=10, cmap=plt_dark,
            marker='*')  # 测试数据
plt.scatter(x_train.T[0], x_train.T[1], c=y_train.ravel(), edgecolors='k', s=40, cmap=plt_dark)  # 全部数据
plt.xlabel(u'特征属性1', fontsize=15)
plt.ylabel(u'特征属性2', fontsize=15)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.grid(True)
plt.title(u'鸢尾花数据的决策树分类', fontsize=18)
plt.show()

# 参数优化
pipe = Pipeline([
    ('mms', MinMaxScaler()),
    ('skb', SelectKBest(chi2)),
    ('pca', PCA()),
    ('decision', DecisionTreeClassifier(random_state=0))
])

# 参数
parameters = {
    "skb__k": [1, 2, 3, 4],
    "pca__n_components": [0.5, 0.99],  # 设置为浮点数代表主成分方差所占最小比例的阈值，这里不建议设置为数值，思考一下？
    "decision__criterion": ["gini", "entropy"],
    "decision__max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}

# 数据
x_train2, x_test2, y_train2, y_test2 = x_train1, x_test1, y_train1, y_test1
# 模型构建；通过网络交叉验证，寻找最优参数列表，param_grid 可选列表参数
gscv = GridSearchCV(pipe, param_grid=parameters, cv=3)
# 模型训练
gscv.fit(x_train2, y_train2)
# 算法的最优解
print("最优参数列表:", gscv.best_params_)
print("score值：", gscv.best_score_)
# 预测值
y_test_hat2 = gscv.predict(x_test2)

# 应用最优参数查看效果
mms_best = MinMaxScaler()
skb_best = SelectKBest(chi2, k=3)
pca_best = PCA(n_components=0.99)
decision_best = DecisionTreeClassifier(criterion='gini', max_depth=4)
# 构建模型并训练模型
x_train3, x_test3, y_train3, y_test3 = x_train1, x_test1, y_train1, y_test1
x_train3 = pca_best.fit_transform(skb_best.fit_transform(mms_best.fit_transform(x_train3), y_train3))
x_test3 = pca_best.transform(skb_best.transform(mms_best.transform(x_test3)))
decision_best.fit(x_train3, y_train3)

print("正确率:", decision_best.score(x_test3, y_test3))

# 基于原始数据前3列比较一下决策树在不同深度的情况下错误率
x_train4, x_test4, y_train4, y_test4 = train_test_split(x.iloc[:, :2], y, train_size=0.7, random_state=14)

depths = np.arange(1, 15)
err_list = []
for d in depths:
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=d)  # 仅设置了这二个参数，没有对数据进行特征选择和降维，所以跟前面得到的结果不同
    clf.fit(x_train4, y_train4)

    score = clf.score(x_test4, y_test4)
    err = 1 - score
    err_list.append(err)
    print("%d深度，正确率%.5f" % (d, score))

## 画图
plt.figure(facecolor='w')
plt.plot(depths, err_list, 'ro-', lw=3)
plt.xlabel(u'决策树深度', fontsize=16)
plt.ylabel(u'错误率', fontsize=16)
plt.grid(True)
plt.title(u'决策树层次太多导致的拟合问题(欠拟合和过拟合)', fontsize=18)
plt.show()

# GridSearchCV 模型保存和加载
from sklearn.externals import joblib

best_gcsv_model = gscv.best_estimator_
joblib.dump(best_gcsv_model, 'gscv.model')
best_gcsv_model2 = joblib.load('gscv.model')
print(best_gcsv_model2.predict(x_test2))
print(y_test_hat2)

# 方式一：输出形成dot文件，然后使用graphviz的dot命令将dot文件转换为pdf
from sklearn import tree

with open('iris.dot', 'w') as f:
    f = tree.export_graphviz(model, out_file=f)
# 命令行执行dot命令： dot -Tpdf iris.dot -o iris.pdf

# 方式二：直接使用pydotplus插件生成pdf文件
from sklearn import tree
import pydotplus

dot_data = tree.export_graphviz(model, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("iris2.pdf")
# graph.write_png("0.png")

# 方式三：直接生成图片
from sklearn import tree
from IPython.display import Image
import pydotplus

dot_data = tree.export_graphviz(model, out_file=None,
                                feature_names=['sepal length', 'sepal width', 'petal length', 'petal width'],
                                class_names=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
                                filled=True, rounded=True,
                                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
