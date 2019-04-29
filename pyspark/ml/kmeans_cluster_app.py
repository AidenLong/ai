# -- encoding:utf-8 --
"""
Create by ibf on 2018/6/22
"""

import os
import time
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
        .setMaster('local[10]') \
        .setAppName('boston housing app')
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sparkContext=sc)

    # 2. 读取数据形成DataFrame
    kmeans_rdd = sc.textFile('../datas/kmeans_data.txt')
    row_kmeans_rdd = kmeans_rdd.map(lambda line: re.split('\\s+', line.strip())) \
        .filter(lambda arr: len(arr) == 3) \
        .map(lambda arr: Row(f1=float(arr[0]), f2=float(arr[1]), f3=float(arr[2])))
    kmeans_df = sqlContext.createDataFrame(row_kmeans_rdd)
    print("数据对应的Schema信息为：{}".format(kmeans_df.schema))
    kmeans_df.show(truncate=False)

    # 3. 特征工程
    # a. 数据合并
    vector = VectorAssembler(inputCols=['f1', 'f2', 'f3'], outputCol='features')
    kmeans_df_tmp01 = vector.transform(kmeans_df)
    kmeans_df_tmp01.cache()

    # 4. 算法构建
    # k：聚类中心数目
    # initMode: 给定聚类中心点的初始化方式，可选参数：random和k-means||
    algo = KMeans(featuresCol="features", predictionCol="prediction", k=2,
                 initMode="k-means||", initSteps=5, tol=1e-4, maxIter=20, seed=28)

    # 5. 算法模型训练
    algo_model = algo.fit(kmeans_df_tmp01)

    # 6. 算法模型预测数据
    result_df = algo_model.transform(kmeans_df_tmp01)
    result_df.show(truncate=False)
    print("聚类中心点:{}".format(algo_model.clusterCenters()))
