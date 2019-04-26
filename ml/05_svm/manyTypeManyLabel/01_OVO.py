# -*- coding:utf-8 -*-
from sklearn import datasets
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC

# 加载数据
iris = datasets.load_iris()

# 获取X和y
X, y = iris.data, iris.target
print("样本数量:%d, 特征数量:%d" % X.shape)

# 模型构建
clf = OneVsOneClassifier(LinearSVC(random_state=0))
# 模型训练
clf.fit(X, y)

# 输出预测结果值
print(clf.predict(X))
print(clf.score(X, y))

# 模型属性输出
k = 1
for item in clf.estimators_:
    print("第%d个模型:" % k, end="")
    print(item)
    k += 1
print(clf.classes_)
