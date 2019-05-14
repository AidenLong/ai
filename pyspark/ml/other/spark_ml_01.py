# --encoding:utf-8 --

import os
import numpy as np
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Matrices
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark import SparkConf, SparkContext

if 'SPARK_HOME' not in os.environ:
    os.environ['SPARK_HOME'] = 'D:\syl\dev\spark-1.6.1-bin-2.5.0-cdh5.3.6'

# 1. 稠密本地向量直接使用numpy中的array或者python中的list表示
dv1 = np.array([1.0, 2.0, 3.0])
dv2 = [4.0, 5.0, 6.0]

# 2. 稀疏本地向量使用pyspark中的Vectors来构建，使用Vector表示
# 表示构建一个3维的向量，下标为0的对应值为1.0，下标为2的对应值为3.0
sv1 = Vectors.sparse(3, [0, 2], [1.0, 3.0])
sv2 = Vectors.sparse(3, {0: 1.0, 2: 3.0})
print(sv1)
print(sv2.toArray())

# 3. 样本标签构建
# 一般我们认为y=1是积极的样本（正例），y=0是消极样本(负例)
positive_label = LabeledPoint(1.0, dv1)
negative_label = LabeledPoint(0.0, sv1)
print(positive_label)
print("样本的特征为:{}, 标签为:{}".format(positive_label.features, positive_label.label))
print(negative_label)

# 4. 本地矩阵(稠密和稀疏两种)； 一般稀疏矩阵基本上不用
# 创建一个3行2列的矩阵（构建的时候从上往下、从左往右进行数据的构建）
dm = Matrices.dense(3, 2, [1, 2, 3, 4, 5, 6])
print(dm)

# 创建Spark上下文
conf = SparkConf() \
    .setMaster('local') \
    .setAppName('spark ml 01')
sc = SparkContext(conf=conf)

# 5. 基于Vector的分布式矩阵
row_vector_rdd = sc.parallelize([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12]
])
mat = RowMatrix(row_vector_rdd)
print("行数:%d" % mat.numRows())
print("列数:%d" % mat.numCols())
print("数据:{}".format(mat.rows.collect()))

