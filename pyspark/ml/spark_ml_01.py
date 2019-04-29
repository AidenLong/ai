# -- encoding:utf-8 --
import os
from pyspark.mllib.stat import Statistics
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, Row

# 给定SPARK_HOME环境变量
if 'SPARK_HOME' not in os.environ:
    os.environ['SPARK_HOME'] = 'D:\syl\dev\spark-1.6.1-bin-2.5.0-cdh5.3.6'

# 1. 创建上下文
# setMaster: 在开发阶段，必须给定，而且只能是：local、local[*]、local[K]；在集群中运行的时候，是通过命令参数给定，代码中最好注释掉
# setAppName：给定应用名称，必须给定
conf = SparkConf() \
    .setMaster('local') \
    .setAppName('spark ml01')
sc = SparkContext(conf=conf)
sql_context = SQLContext(sparkContext=sc)

# 2. 构建RDD
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
print("样本的数据量:{}".format(summary.count()))
print("各个特征属性的最大值:{}".format(summary.max()))
print("各个特征属性的最小值:{}".format(summary.min()))
print("各个特征属性上值不为0的样本数目:{}".format(summary.numNonzeros()))
print("各个特征属性的L1范式值:{}".format(summary.normL1()))
print("各个特征属性的L2范式值:{}".format(summary.normL2()))

# 2. 相关性计算
x = sc.parallelize([1.0, 1.5, 2.0, 2.5, 0.9, 0.86, 0.85, 0.1])
y = sc.parallelize([10.0, 14.0, 19.0, 30.0, 11.0, 9.5, 9.4, 0.0])
print("x和y的相关性值为:{}".format(Statistics.corr(x, y)))
# 这里计算的是特征与特征之间的相关性，也就是输入的rdd矩阵中的列与列之间的相关性
# method; 给定相关性的一个度量方式，默认为皮尔逊相似度(pearson)，可选spearman(斯皮尔曼相关系系数)
feature_corr = Statistics.corr(vector_rdd, method='pearson')
print("RDD中各个特征属性之间的相关性值为:\n{}".format(feature_corr))
