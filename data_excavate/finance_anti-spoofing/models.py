# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel

# 设置字符集，防止图片中的中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

# 特征之后的数据读取
data = pd.read_csv("./data/features01.csv")
print(data.head(3))

# 获取X和Y
Y = data['loan_status']
X = data.drop(['loan_status'], 1, inplace=False)
print("样本数量为:%d, 特征属性数量为:%d" % X.shape)

# 样本的分割
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
print("训练集合数据量:%d,%d" % x_train.shape)
print("测试集合数据量:%d,%d" % x_test.shape)

# 一般情况下：在做分类的时候，都会看一下各个类别的样本数量的比例，看一下是否存在数据的不平衡情况
print(y_train.value_counts())
print(y_test.value_counts())

# 首先做一个最优参数的构造
parameters = {
    "penalty": ['l1', 'l2'],
    "C": [0.01, 0.1, 1],
    "fit_intercept": [True, False],
    "max_iter": [100, 150, 200]
}
clf = GridSearchCV(LogisticRegression(random_state=0), param_grid=parameters, cv=3)
clf.fit(x_train, y_train)

# 得到最优参数
print("最优参数:", end="")
print(clf.best_params_)

# 使用逻辑回归来分析数据
lr = LogisticRegression(C=0.1, fit_intercept=True, max_iter=100, penalty='l1', random_state=0)
lr.fit(x_train, y_train)

train_predict = lr.predict(x_train)
print("训练集合上的f1指标:%.4f" % f1_score(y_train, train_predict))
test_predict = lr.predict(x_test)
print("测试集合上的f1指标:%.4f" % f1_score(y_test, test_predict))

# 使用逻辑回归来分析数据 + 可以选择给类别添加权重
# 加入权重后，模型效果变的更差：原因可能是，两个类别之间的比例没有那么悬殊或者数据上来讲两个类别的数据融合在一起的
weight = {
    0: 5,  # 在模型训练和测试的过程中，类别0的重要性
    1: 1  # 在模型训练和测试的过程中，类别1的重要性
}
lr = LogisticRegression(C=0.1,
                        fit_intercept=True,
                        max_iter=100,
                        penalty='l1',
                        random_state=0,
                        class_weight=weight
                        )
lr.fit(x_train, y_train)

train_predict = lr.predict(x_train)
print("训练集合上的f1指标:%.4f" % f1_score(y_train, train_predict))
test_predict = lr.predict(x_test)
print("测试集合上的f1指标:%.4f" % f1_score(y_test, test_predict))

# 使用随机森林来分析数据
forest = RandomForestClassifier(random_state=0, max_depth=5)
forest.fit(x_train, y_train)

train_predict = forest.predict(x_train)
print("训练集合上的f1指标:%.4f" % f1_score(y_train, train_predict))
test_predict = forest.predict(x_test)
print("测试集合上的f1指标:%.4f" % f1_score(y_test, test_predict))

# 基于随机森林获取影响放贷的二十大因素
feature_importances = forest.feature_importances_
feature_importances = 100.0 * (feature_importances / feature_importances.max())

indices = np.argsort(feature_importances)[-20:]
plt.barh(np.arange(20), feature_importances[indices], color='dodgerblue', alpha=0.4)
plt.yticks(np.arange(20 + 0.25), np.array(X.columns)[indices])
plt.xlabel('特征重要性百分比')
plt.title('随机森林20大重要特征提取')
plt.show()

# 用随机森林选择特征，然后使用Logistic回归来做预测
# a. 特征选择过程
print("原始样本大小:{}".format(x_train.shape))
forest = RandomForestClassifier(random_state=0, max_depth=5)
# 当特征的权重大于等于给定的threshold的时候，该特征就保留；由于随机森林中的特征属性权重一定是大于等于0的值，所以一般情况下，，
# 在决策树类型的算法中，使用SelectFromModel一般选择比0稍大一点点的阈值。
sm = SelectFromModel(estimator=forest, threshold=0.0000001)
sm.fit(x_train, y_train)
x_train1 = sm.transform(x_train)
x_test1 = sm.transform(x_test)
print("原始样本大小:{}".format(x_train1.shape))
# b. logistic回归训练
lr = LogisticRegression(C=0.1, fit_intercept=True, max_iter=100, penalty='l1', random_state=0)
lr.fit(x_train1, y_train)

train_predict = lr.predict(x_train1)
print("训练集合上的f1指标:%.4f" % f1_score(y_train, train_predict))
test_predict = lr.predict(x_test1)
print("测试集合上的f1指标:%.4f" % f1_score(y_test, test_predict))

# GBDT的提取的效果
gbdt = GradientBoostingClassifier(min_samples_split=50, max_depth=2, n_estimators=300, learning_rate=0.1,
                                  random_state=0)
gbdt.fit(x_train, y_train)

train_predict = gbdt.predict(x_train)
print("训练集合上的f1指标:%.4f" % f1_score(y_train, train_predict))
test_predict = gbdt.predict(x_test)
print("测试集合上的f1指标:%.4f" % f1_score(y_test, test_predict))

# 基于GBDT获取影响放贷的二十大因素
feature_importances = gbdt.feature_importances_
feature_importances = 100.0 * (feature_importances / feature_importances.max())

indices = np.argsort(feature_importances)[-20:]
plt.barh(np.arange(20), feature_importances[indices], color='dodgerblue', alpha=0.4)
plt.yticks(np.arange(20 + 0.25), np.array(X.columns)[indices])
plt.xlabel('特征重要性百分比')
plt.title('GBDT 20大重要特征提取')
plt.show()

print("权重大于0的特征数目:{}".format(np.sum(gbdt.feature_importances_ > 0)))
print("权重等于0的特征数目:{}".format(np.sum(gbdt.feature_importances_ == 0)))
print("权重小于0的特征数目:{}".format(np.sum(gbdt.feature_importances_ < 0)))

# 在实际工作中，如果发现模型的效果不如意，那么可能需要考虑特征选择和降维
# 使用逻辑回归来分析数据 + 特征选择 + 降维
# 特征选择：从所有特征属性中抽取出来影响目标属性(target)效果最大的特征属性作为下一步的特征属性列表\
# 很多特征选择工程都是选择方差比较大特征属性
# 也可以使用随机森林、GBDT、决策树来进行特征选择

# 降维：压缩样本的维度空间，直白来讲，就是讲DataFrame中原本的多个列合并成为一列

# 1. 特征选择
feature_importances = gbdt.feature_importances_
indices = np.argsort(feature_importances)[-30:]
top30_features = np.array(X.columns)[indices]

# 2. 提取影响最大的三十个特征属性
x_train2 = x_train[top30_features]
x_test2 = x_test[top30_features]

# 3. 降维处理(在三十个特征之外，将其它的特征数据做一个降维操作)
x_train3 = x_train.drop(top30_features, 1, inplace=False)
x_test3 = x_test.drop(top30_features, 1, inplace=False)
pca = PCA(n_components=10)
pca.fit(x_train3)
x_test3 = pca.transform(x_test3)
x_train3 = pca.transform(x_train3)

# 4. 两个DataFrame合并
x_train2 = np.hstack([x_train2, x_train3])
x_test2 = np.hstack([x_test2, x_test3])
print("合并后样本大小:{}".format(x_train2.shape))

lr = LogisticRegression(C=0.1, fit_intercept=True, max_iter=100, penalty='l1', random_state=0)
lr.fit(x_train2, y_train)

train_predict = lr.predict(x_train2)
print("训练集合上的f1指标:%.4f" % f1_score(y_train, train_predict))
test_predict = lr.predict(x_test2)
print("测试集合上的f1指标:%.4f" % f1_score(y_test, test_predict))


# 计算KS的方式一：
def compute_ks(data):
    sorted_list = data.sort_values(['predict_proba'], ascending=[True])  # 按照样本为正样本的概率值升序排序 ，也即坏样本的概率从高到低排序
    total_good = sorted_list['label'].sum()
    total_bad = sorted_list.shape[0] - total_good
    max_ks = 0.0
    good_count = 0.0
    bad_count = 0.0
    for index, row in sorted_list.iterrows():  # 按照标签和每行拆开
        if row['label'] == 0:
            bad_count += 1
        else:
            good_count += 1
        val = abs(bad_count / total_bad - good_count / total_good)
        max_ks = max(max_ks, val)
    return max_ks


test_pd = pd.DataFrame()
y_predict_proba = lr.predict_proba(x_test2)[:, 1]  # 取被分为正样本的概率那一列
Y_test_1 = np.array(y_test)
test_pd['label'] = Y_test_1
test_pd['predict_proba'] = y_predict_proba
print("测试集 KS:", compute_ks(test_pd))

# 计算KS的方式二
y_predict_proba = lr.predict_proba(x_test2)[:, 1]
fpr, tpr, thresholds = sklearn.metrics.roc_curve(np.array(y_test), y_predict_proba)
print('KS:', max(tpr - fpr))
