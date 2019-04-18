# -*- coding:utf-8 -*-
from sklearn.neighbors import KDTree
import numpy as np

np.random.seed(0)
X = np.random.random((10, 3))
print(X)

tree = KDTree(X, leaf_size=2)
print(tree)

dist, bind = tree.query([X[0]], k=3)
print(dist)
print(bind)
