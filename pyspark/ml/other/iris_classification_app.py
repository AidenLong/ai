# --encoding:utf-8 --

import os
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, Row
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier, MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

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
    iris_rdd = sc.textFile("../datas/iris.data")
    row_iris_rdd = iris_rdd.map(lambda line: line.split(',')) \
        .filter(lambda arr: len(arr) == 5) \
        .map(lambda arr: Row(f1=float(arr[0]), f2=float(arr[1]), \
                             f3=float(arr[2]), f4=float(arr[3]), \
                             f5=arr[4]))
    row_iris_df = sql_context.createDataFrame(row_iris_rdd)
    print("原始数据对应的schema信息为:", end='')
    print(row_iris_df.schema)
    row_iris_df.show(truncate=False)

    # 0. 数据划分为训练集和测试集
    train_df, test_df = row_iris_df.randomSplit(weights=[0.7, 0.3], seed=28)

    # 1. 特征工程
    # a. label标签数值型化
    indexer = StringIndexer(inputCol='f5', outputCol='label', handleInvalid='error')
    indexer_model = indexer.fit(train_df)
    train_df_tmp01 = indexer_model.transform(train_df)
    test_df_tmp01 = indexer_model.transform(test_df)
    # b. 特征属性合并
    vector = VectorAssembler(inputCols=['f1', 'f2', 'f3', 'f4'], outputCol='features')
    train_df_tmp02 = vector.transform(train_df_tmp01)
    test_df_tmp02 = vector.transform(test_df_tmp01)

    # 缓存数据（因为算法构建式一个迭代类型的，表示对于数据而言需要多次的读取，所以这里最好将数据进行缓存，从而保证后续算法模型构建的性能/执行速度）
    train_df_tmp02.cache()

    # 2. 算法模型
    tree = DecisionTreeClassifier(featuresCol='features', labelCol='label', predictionCol='prediction', maxDepth=5)
    tree_model = tree.fit(train_df_tmp02)
    # 使用训练好的模型对数据进行转换/预测操作
    tree_model.transform(test_df_tmp02).show(truncate=False)


    # 使用多层感知器模型
    # layers：给定列表，列表中给定每一层的感知器的数目, 第一个是给定输入的维度大小，最后一个是给定输出的维度大小（可以认为是y值的类别数目）
    mpc = MultilayerPerceptronClassifier(featuresCol="features", labelCol="label", \
                                         predictionCol="prediction", maxIter=100, \
                                         seed=28, layers=[4, 8, 4, 5, 3])
    mpc_model = mpc.fit(train_df_tmp02)
    # 使用训练好的模型进行数据的预测
    mpc_model.transform(test_df_tmp02).show(truncate=False)

    name, model = "多层感知器模型", mpc_model
    name, model = "决策树", tree_model
    # 3. 模型效果评估
    # metricName: 给定使用何种评估指标，可选参数："f1", "precision", "recall", "weightedPrecision", "weightedRecall"
    evaluator = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='label', metricName='precision')
    print("{}在训练集上精确率={}".format(name, evaluator.evaluate(model.transform(train_df_tmp02))))
    print("{}在测试集上精确率={}".format(name, evaluator.evaluate(model.transform(test_df_tmp02))))
