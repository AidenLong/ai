# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv('./data/merge_user_item_rating.csv')
data = data.drop(['user_id', 'item_id'], axis=1)
# print(data.head())
# print(data.info())

Y = data['rating'].astype(np.int64)
X = data.drop('rating', axis=1)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.05, random_state=28)
print("训练集合数据量:%d, %d" % x_train.shape)
print("测试集合数据量:%d，%d" % x_test.shape)

model = DecisionTreeClassifier(max_depth=50)
# model = KNeighborsClassifier(n_neighbors=10)
# model = SVC(C=1.0, kernel='rbf')
model.fit(x_train, y_train)

print('训练集准确率:{}'.format(model.score(x_train, y_train)))
print('测试集准确率:{}'.format(model.score(x_test, y_test)))

tmp_x = [
    [21, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0]
]
tmp_y = model.predict(tmp_x)
print("预测值为:{}".format(tmp_y))

