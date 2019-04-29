# -- encoding:utf-8 --
"""
Create by ibf on 2018/6/22
"""

import os
import time
import re
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, Row
from pyspark.ml.feature import PolynomialExpansion, VectorAssembler, PCA
from pyspark.ml.regression import LinearRegression, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

if 'SPARK_HOME' not in os.environ:
    os.environ['SPARK_HOME'] = 'D:\syl\dev\spark-1.6.1-bin-2.5.0-cdh5.3.6'

if __name__ == '__main__':
    name = '线性回归'
    # name = 'GBDT'

    # 1. 上下文构建
    conf = SparkConf() \
        .setMaster('local[10]') \
        .setAppName('boston housing app')
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sparkContext=sc)

    # 2. 读取数据形成DataFrame
    boston_housing_rdd = sc.textFile('../datas/boston_housing.data')
    row_boston_housing_rdd = boston_housing_rdd.map(lambda line: re.split('\\s+', line.strip())) \
        .filter(lambda arr: len(arr) == 14) \
        .map(lambda arr: Row(CRIM=float(arr[0]), ZN=float(arr[1]), \
                             INDUS=float(arr[2]), CHAS=float(arr[3]), NOX=float(arr[4]), \
                             RM=float(arr[5]), AGE=float(arr[6]), DIS=float(arr[7]), \
                             RAD=float(arr[8]), TAX=float(arr[9]), PTRATIO=float(arr[10]), \
                             B=float(arr[11]), LASTAT=float(arr[12]), MEDV=float(arr[13])))
    boston_housing_df = sqlContext.createDataFrame(row_boston_housing_rdd)
    print("数据对应的Schema信息为：{}".format(boston_housing_df.schema))
    boston_housing_df.show(truncate=False)

    # 3. 特征工程
    # a. 将数据分割为训练集和测试集
    train_df, test_df = boston_housing_df.randomSplit(weights=[0.7, 0.3], seed=28)

    # b. 合并特征
    input_all_feature_names = boston_housing_df.schema.names
    input_all_feature_names.remove('MEDV')
    vector = VectorAssembler(inputCols=input_all_feature_names, outputCol='f1')
    train_df_tmp01 = vector.transform(train_df)
    test_df_tmp01 = vector.transform(test_df)

    # c. 多项式扩展
    poly = PolynomialExpansion(degree=3, inputCol='f1', outputCol='f2')
    train_df_tmp02 = poly.transform(train_df_tmp01)
    test_df_tmp02 = poly.transform(test_df_tmp01)

    # d. 降维
    pca = PCA(k=75, inputCol='f2', outputCol='features')
    pca_model = pca.fit(train_df_tmp02)
    train_df_tmp03 = pca_model.transform(train_df_tmp02)
    test_df_tmp03 = pca_model.transform(test_df_tmp02)

    # 4. 模型训练&结果预测
    # a. 因为模型训练其实是一个迭代的过程，所以数据最好缓存
    train_df_tmp03.cache()

    # b. 模型构建
    if name == '线性回归':
        """
        featuresCol="features", 特征属性所对应的列名称
        labelCol="label", 标签y对应的列名称
        predictionCol="prediction", 模型预测值对应的列名称，要求模型训练前该列不存在
        maxIter=100, 迭代次数
        regParam=0.0, 惩罚项的学习率或者学习因子，当设置为0的时候，表示不使用惩罚性/正则项，计算公式: regParam * ((1-p)*L2 + p*L1)
        elasticNetParam=0.0, 给定EN弹性网络中，p的值。当设置为0的时候，表示使用L2正则，此时算法即：Ridge，当设置为1的时候，表示只使用L1正则，此时算法即：Lasso
        tol=1e-6,
        fitIntercept=True, 模型训练中，是否训练截距项，True表示训练
        standardization=True, 在模型训练之前是否做一个数据的标注化操作，默认为True，表示做。
        solver="auto", 算法模型的底层求解的方式方法
        weightCol=None 各个样本的权重所对应的列名称，可选。
        """
        algo = LinearRegression(featuresCol='features', labelCol='MEDV', predictionCol='prediction', maxIter=100,
                                regParam=0.1, elasticNetParam=0.0)
    elif name == 'GBDT':
        algo = GBTRegressor(featuresCol="features", labelCol="MEDV", predictionCol="prediction", maxDepth=2,
                            maxIter=100, subsamplingRate=0.8, stepSize=0.1)
    else:
        raise Exception("现在的代码不支持该算法:{}".format(name))

    # c. 模型训练
    algo_model = algo.fit(train_df_tmp03)

    # d. 训练好后对数据做一个预测（预测结果可以通过DataFrame或者RDD的相关API进行输出）
    train_predict_result_df = algo_model.transform(train_df_tmp03)
    train_predict_result_df.select('MEDV', 'prediction').show(truncate=False)
    test_predict_result_df = algo_model.transform(test_df_tmp03)
    test_predict_result_df.select('MEDV', 'prediction').show(truncate=False)

    # 5. 模型效果评估
    # metricName：给定采用何种评估指标，默认为rmse；可选参数：mse|rmse|r2|mae
    evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="MEDV", metricName="r2")
    print("{}算法在训练集上的R2值为:{}".format(name, evaluator.evaluate(train_predict_result_df)))
    print("{}算法在测试集上的R2值为:{}".format(name, evaluator.evaluate(test_predict_result_df)))

    # 为了看一下4040界面，休息一下
    time.sleep(60)
