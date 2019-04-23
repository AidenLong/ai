# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import label_binarize
from sklearn import metrics

mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

names = [u'Age', u'Number of sexual partners', u'First sexual intercourse',
         u'Num of pregnancies', u'Smokes', u'Smokes (years)',
         u'Smokes (packs/year)', u'Hormonal Contraceptives',
         u'Hormonal Contraceptives (years)', u'IUD', u'IUD (years)', u'STDs',
         u'STDs (number)', u'STDs:condylomatosis',
         u'STDs:cervical condylomatosis', u'STDs:vaginal condylomatosis',
         u'STDs:vulvo-perineal condylomatosis', u'STDs:syphilis',
         u'STDs:pelvic inflammatory disease', u'STDs:genital herpes',
         u'STDs:molluscum contagiosum', u'STDs:AIDS', u'STDs:HIV',
         u'STDs:Hepatitis B', u'STDs:HPV', u'STDs: Number of diagnosis',
         u'STDs: Time since first diagnosis', u'STDs: Time since last diagnosis',
         u'Dx:Cancer', u'Dx:CIN', u'Dx:HPV', u'Dx', u'Hinselmann', u'Schiller',
         u'Citology', u'Biopsy']  # df.columns
path = "datas/risk_factors_cervical_cancer.csv"  # 数据文件路径
data = pd.read_csv(path)

X = data[names[0:-4]]
Y = data[names[-4:]]
print(X.head(1))  # 随机森林可以处理多个目标变量的情况
print(Y.head(1))

# 空值的处理
X = X.replace("?", np.NAN)
# 使用Imputer给定缺省值，默认的是以mean
# 对于缺省值，进行数据填充；默认是以列/特征的均值填充
imputer = SimpleImputer(missing_values=np.nan)
X = imputer.fit_transform(X, Y)

# 数据分割
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
print("训练样本数量:%d,特征属性数目:%d,目标属性数目:%d" % (x_train.shape[0], x_train.shape[1], y_train.shape[1]))
print("测试样本数量:%d" % x_test.shape[0])

# 标准化
ss = MinMaxScaler()  # 分类模型，经常使用的是minmaxscaler归一化，回归模型经常用standardscaler
x_train = ss.fit_transform(x_train, y_train)
x_test = ss.transform(x_test)
print(x_train.shape)

# 降维
pca = PCA(n_components=2)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
print(x_train.shape)
print(pca.explained_variance_ratio_)

# 随机森林模型
forest = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=1, random_state=0)
forest.fit(x_train, y_train)  # max_depth一般不宜设置过大，把每个模型作为一个弱分类器

# 模型效果评估
score = forest.score(x_test, y_test)
print("准确率:%.2f%%" % (score * 100))
# 模型预测
forest_y_score = forest.predict_proba(x_test)  # prodict_proba输出概率

# 模型效果评估
score = forest.score(x_test, y_test)
print('准确率：%.2f%%' % (score * 100))
# 模型预测
forest_y_score = forest.predict_proba(x_test)
# 计算ROC值
forest_fpr1, forest_tpr1, _ = metrics.roc_curve(label_binarize(y_test[names[-4]], classes=(0, 1, 2)).T[0:-1].T.ravel(),
                                                forest_y_score[0].ravel())
forest_fpr2, forest_tpr2, _ = metrics.roc_curve(label_binarize(y_test[names[-3]], classes=(0, 1, 2)).T[0:-1].T.ravel(),
                                                forest_y_score[1].ravel())
forest_fpr3, forest_tpr3, _ = metrics.roc_curve(label_binarize(y_test[names[-2]], classes=(0, 1, 2)).T[0:-1].T.ravel(),
                                                forest_y_score[2].ravel())
forest_fpr4, forest_tpr4, _ = metrics.roc_curve(label_binarize(y_test[names[-4]], classes=(0, 1, 2)).T[0:-1].T.ravel(),
                                                forest_y_score[3].ravel())
# AUC值
auc1 = metrics.auc(forest_fpr1, forest_tpr1)
auc2 = metrics.auc(forest_fpr2, forest_tpr2)
auc3 = metrics.auc(forest_fpr3, forest_tpr3)
auc4 = metrics.auc(forest_fpr4, forest_tpr4)

print("Hinselmann目标属性AUC值：", auc1)
print("Schiller目标属性AUC值：", auc2)
print("Citology目标属性AUC值：", auc3)
print("Biopsy目标属性AUC值：", auc4)

# 画图
plt.figure(figsize=(8, 6), facecolor='w')
plt.plot(forest_fpr1, forest_tpr1, c='r', lw=2, label=u'Hinselmann目标属性,AUC=%.3f' % auc1)
plt.plot(forest_fpr2, forest_tpr2, c='b', lw=2, label=u'Schiller目标属性,AUC=%.3f' % auc2)
plt.plot(forest_fpr3, forest_tpr3, c='g', lw=2, label=u'Citology目标属性,AUC=%.3f' % auc3)
plt.plot(forest_fpr4, forest_tpr4, c='y', lw=2, label=u'Biopsy目标属性,AUC=%.3f' % auc4)
plt.plot((0, 1), (0, 1), c='#a0a0a0', lw=2, ls='--')
plt.xlim(-0.001, 1.001)
plt.ylim(-0.001, 1.001)
plt.xticks(np.arange(0, 1.1, 0.1))
plt.xticks(np.arange(0, 1.1, 0.1))
plt.xlabel('False Positive Rate(FPR)', fontsize=16)
plt.ylabel('True Positive Rate(TPR)', fontsize=16)
plt.grid(b=True, ls=':')
plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
plt.title(u'随机森林多目标属性分类ROC曲线', fontsize=18)
plt.show()

# 比较不同树数目、树最大深度的情况下随机森林的正确率
# 一般情况下，初始的随机森林树个数是100，深度1，如果需要我们再进行优化操作
x_train2, x_test2, y_train2, y_test2 = train_test_split(X, Y, test_size=0.5, random_state=0)
print("训练样本数量%d，测试样本数量:%d" % (x_train2.shape[0], x_test2.shape[0]))
# 比较
estimators = [1, 50, 100, 500]
depth = [1, 2, 3, 7, 15]
x1, x2 = np.meshgrid(estimators, depth)
err_list = []
for es in estimators:
    es_list = []
    for d in depth:
        tf = RandomForestClassifier(n_estimators=es, criterion='gini', max_depth=d, max_features=None, random_state=0)
        tf.fit(x_train2, y_train2)
        st = tf.score(x_test2, y_test2)
        err = 1 - st
        es_list.append(err)
        print("%d决策树数目，%d最大深度，正确率:%.2f%%" % (es, d, st * 100))
    err_list.append(es_list)

# 画图
plt.figure(facecolor='w')
i = 0
colors = ['r', 'b', 'g', 'y']
lw = [1, 2, 4, 3]
max_err = 0
min_err = 100
for es, l in zip(estimators, err_list):
    plt.plot(depth, l, c=colors[i], lw=lw[i], label=u'树数目:%d' % es)
    max_err = max((max(l), max_err))
    min_err = min((min(l), min_err))
    i += 1
plt.xlabel(u'树深度', fontsize=16)
plt.ylabel(u'错误率', fontsize=16)
plt.legend(loc='upper left', fancybox=True, framealpha=0.8, fontsize=12)
plt.grid(True)
plt.xlim(min(depth), max(depth))
plt.ylim(min_err * 0.99, max_err * 1.01)
plt.title(u'随机森林中树数目、深度和错误率的关系图', fontsize=18)
plt.show()

# 随机森林画图
# 方式三：直接生成图片
from sklearn import tree
import pydotplus

k = 0
for clf in forest.estimators_:
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf("foress_tree_%d.pdf" % k)
    k += 1
