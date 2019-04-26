# -*- coding:utf-8 -*-
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# 数据获取
iris = datasets.load_iris()
X, y = iris.data, iris.target
print("样本数量:%d, 特征数量:%d" % X.shape)

# 模型创建
clf = OneVsRestClassifier(LinearSVC(random_state=0))
# 模型构建
clf.fit(X, y)

# 预测结果输出
print(clf.predict(X))
print(clf.score(X, y))

# 模型属性输出
k = 1
for item in clf.estimators_:
    print("第%d个模型:" % k, end="")
    print(item)
    k += 1
print(clf.classes_)
