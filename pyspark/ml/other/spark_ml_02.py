# --encoding:utf-8 --

import os
from pyspark.mllib.stat import Statistics
from pyspark import SparkConf, SparkContext

if 'SPARK_HOME' not in os.environ:
    os.environ['SPARK_HOME'] = 'D:\syl\dev\spark-1.6.1-bin-2.5.0-cdh5.3.6'

# 创建Spark上下文
conf = SparkConf() \
    .setMaster('local') \
    .setAppName('spark ml 02')
sc = SparkContext(conf=conf)

# 构建一个Vector的RDD
vector_rdd = sc.parallelize([
    [0, 2, 3],
    [4, 8, 16],
    [-7, 8, -9],
    [10, -10, 12]
])

# 1. 汇总统计
summary = Statistics.colStats(vector_rdd)
print("汇总对象类型:{}".format(type(summary)))
print("各个特征属性的均值:{}".format(summary.mean()))
print("各个特征属性的方差:{}".format(summary.variance()))
print("样本数据量:{}".format(summary.count()))
print("各个特征属性的最大特征值:{}".format(summary.max()))
print("各个特征属性的最小特征值:{}".format(summary.min()))
print("特征值不等于0的样本数量:{}".format(summary.numNonzeros()))
print("各个特征的L1范式值:{}".format(summary.normL1()))
print("各个特征的L2范式值:{}".format(summary.normL2()))

# 2. 相关性统计(特征与特征之间的相关性统计)
x = sc.parallelize([1.0, 1.5, 0.9, 0, 0.85, 0.95, 0.5])
y = sc.parallelize([2.0, 2.1, 0, 2.0, 0, 2.21, 0])
print("x和y的相关性指标值为:{}".format(Statistics.corr(x, y)))
# method给定相关性计算方式，默认为pearson(皮尔逊相关系数)，另外可选:spearman(斯皮尔曼相关性系数)
feature_corr = Statistics.corr(vector_rdd, method='pearson')
print("RDD对象中特征属性与特征属性之间的相关性指标为:\n{}".format(feature_corr))
