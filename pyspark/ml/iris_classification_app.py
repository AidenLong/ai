# -- encoding:utf-8 --
"""
鸢尾花分类的应用
Create by ibf on 2018/6/22
"""

import os
import time
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, Row
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier, MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

if 'SPARK_HOME' not in os.environ:
    os.environ['SPARK_HOME'] = 'D:\syl\dev\spark-1.6.1-bin-2.5.0-cdh5.3.6'

if __name__ == '__main__':
    name = '决策树'
    name = '多层感知器'

    # 1. 上下文构建
    conf = SparkConf() \
        .setMaster('local[10]') \
        .setAppName('iris app')
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sparkContext=sc)

    # 2. 读取数据形成DataFrame
    iris_rdd = sc.textFile('../datas/iris.data')
    row_iris_rdd = iris_rdd.map(lambda line: line.split(',')) \
        .filter(lambda arr: len(arr) == 5) \
        .map(lambda arr: Row(f1=float(arr[0]), f2=float(arr[1]), f3=float(arr[2]), f4=float(arr[3]), f5=arr[4]))
    iris_df = sqlContext.createDataFrame(row_iris_rdd)
    print("数据对应的Schema信息为：{}".format(iris_df.schema))
    iris_df.show(truncate=False)

    # 3. 特征工程
    # a. 将数据分割为训练集和测试集
    train_df, test_df = iris_df.randomSplit(weights=[0.7, 0.3], seed=28)

    # b. 将预测属性f5的数据类型转换为数值类型
    indexer = StringIndexer(inputCol='f5', outputCol='label', handleInvalid='error')
    indexer_model = indexer.fit(train_df)
    train_df_tmp01 = indexer_model.transform(train_df)
    test_df_tmp01 = indexer_model.transform(test_df)

    # c. 合并特征
    vector = VectorAssembler(inputCols=['f1', 'f2', 'f3', 'f4'], outputCol='features')
    train_df_tmp02 = vector.transform(train_df_tmp01)
    test_df_tmp02 = vector.transform(test_df_tmp01)

    # 4. 模型训练&结果预测
    # a. 因为模型训练其实是一个迭代的过程，所以数据最好缓存
    train_df_tmp02.cache()

    # b. 模型构建
    if name == '决策树':
        algo = DecisionTreeClassifier(featuresCol='features', labelCol='label', predictionCol='prediction', maxDepth=5)
    elif name == '多层感知器':
        # layers: 给定神经网络中，每一层的感知器数目/神经元数目，第一个数字是输入的样本维度大小，最后一个数字是给定的输出类别数目
        algo = MultilayerPerceptronClassifier(featuresCol="features", labelCol="label", predictionCol="prediction",
                                              maxIter=100, tol=1e-4, seed=28, layers=[4, 5, 6, 8, 3])
    else:
        raise Exception("现在的代码不支持该算法:{}".format(name))

    # c. 模型训练
    algo_model = algo.fit(train_df_tmp02)

    # d. 训练好后对数据做一个预测（预测结果可以通过DataFrame或者RDD的相关API进行输出）
    train_predict_result_df = algo_model.transform(train_df_tmp02)
    train_predict_result_df.select('label', 'prediction').show(truncate=False)
    test_predict_result_df = algo_model.transform(test_df_tmp02)
    test_predict_result_df.select('label', 'prediction').show(truncate=False)

    # 5. 模型效果评估
    # metricName：给定采用何种评估指标，默认为f1；可选参数：f1|precision|recall|weightedPrecision|weightedRecall
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label", metricName="f1")
    print("{}算法在训练集上的F1值为:{}".format(name, evaluator.evaluate(train_predict_result_df)))
    print("{}算法在测试集上的F1值为:{}".format(name, evaluator.evaluate(test_predict_result_df)))

    # 为了看一下4040界面，休息一下
    time.sleep(60)
