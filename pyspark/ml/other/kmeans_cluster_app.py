# --encoding:utf-8 --

import os
import re
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, Row
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans

if 'SPARK_HOME' not in os.environ:
    os.environ['SPARK_HOME'] = 'D:\syl\dev\spark-1.6.1-bin-2.5.0-cdh5.3.6'

if __name__ == '__main__':
    # 1. 上下文构建
    conf = SparkConf() \
        .setMaster('local') \
        .setAppName('iris classification app')
    sc = SparkContext(conf=conf)
    sql_context = SQLContext(sparkContext=sc)

    # 2. RDD&DataFrame构建
    kmeans_rdd = sc.textFile("../datas/kmeans_data.txt")
    row_kmeans_rdd = kmeans_rdd.map(lambda line: re.split('\\s+', line.strip())) \
        .filter(lambda arr: len(arr) == 3) \
        .map(lambda arr: Row(f1=float(arr[0]), f2=float(arr[1]), f3=float(arr[2])))
    row_kmeans_df = sql_context.createDataFrame(row_kmeans_rdd)
    print("原始数据对应的schema信息为:", end='')
    print(row_kmeans_df.schema)
    row_kmeans_df.show(truncate=False)

    # 1. 特征工程
    # a. 数据合并
    vector = VectorAssembler(inputCols=['f1', 'f2', 'f3'], outputCol='features')
    tmp01_df = vector.transform(row_kmeans_df)

    # 2. 算法模型
    # initMode: 给定初始点随机选择方式，eg：random和k-means||
    kmeans = KMeans(featuresCol="features", predictionCol="prediction", k=2, \
                    initMode="random", tol=1e-4, maxIter=20, seed=28)
    kmeans_model = kmeans.fit(tmp01_df)

    # 3. 模型预测
    # computeCost： python中不支持这个API
    # print("损失函数的值:{}".format(kmeans_model.computeCost(tmp01_df)))
    print("聚类中心点:{}".format(kmeans_model.clusterCenters()))
    # 预测样本所属的类别
    kmeans_model.transform(tmp01_df).show(truncate=False)
