# -*- coding:utf-8 -*-
import os
import time
import re
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, Row
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.clustering import KMeans

if 'SPARK_HOME' not in os.environ:
    os.environ['SPARK_HOME'] = 'D:\syl\dev\spark-1.6.1-bin-2.5.0-cdh5.3.6'

if __name__ == '__main__':
    # 1. 上下文构建
    conf = SparkConf() \
        .setMaster('local[10]') \
        .setAppName('kmeans app')
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sparkContext=sc)

    kmeans_rdd = sc.textFile("../datas/kmeans_data.txt")
    row_kmeans_rdd = kmeans_rdd.map(lambda line: re.split('\\s+', line.strip())) \
        .filter(lambda arr: len(arr) == 3) \
        .map(lambda arr: Row(f1=float(arr[0]), f2=float(arr[1]), f3=float(arr[2])))

    kmeans_df = sqlContext.createDataFrame(row_kmeans_rdd)
    print(kmeans_df.schema)
    kmeans_df.show(truncate=False)

    vector = VectorAssembler(inputCols=['f1', 'f2', 'f3'], outputCol='features')
    kmeans_df_tmp01 = vector.transform(kmeans_df)
    kmeans_df_tmp01.cache()

    algo = KMeans(featuresCol="features", predictionCol="prediction", k=3, seed=28)

    algo_model = algo.fit(kmeans_df_tmp01)

    result_df = algo_model.transform(kmeans_df_tmp01)
    result_df.show(truncate=False)
    print(algo_model.clusterCenters())
