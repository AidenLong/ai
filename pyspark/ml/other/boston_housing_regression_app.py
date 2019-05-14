# --encoding:utf-8 --

import os
import re
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, Row
from pyspark.ml.feature import PolynomialExpansion, VectorAssembler, PCA
from pyspark.ml.regression import LinearRegression, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

if 'SPARK_HOME' not in os.environ:
    os.environ['SPARK_HOME'] = 'D:\syl\dev\spark-1.6.1-bin-2.5.0-cdh5.3.6'

if __name__ == '__main__':
    # 1. 上下文构建
    conf = SparkConf() \
        .setMaster('local') \
        .setAppName('spark ml 02')
    sc = SparkContext(conf=conf)
    sql_context = SQLContext(sparkContext=sc)

    # 2. RDD&DataFrame构建
    boston_house_rdd = sc.textFile("../datas/boston_housing.data")
    row_boston_house_rdd = boston_house_rdd.map(lambda line: re.split('\\s+', line.strip())) \
        .filter(lambda arr: len(arr) == 14) \
        .map(lambda arr: Row(CRIM=float(arr[0]), ZN=float(arr[1]), \
                             INDUS=float(arr[2]), CHAS=float(arr[3]), NOX=float(arr[4]), \
                             RM=float(arr[5]), AGE=float(arr[6]), DIS=float(arr[7]), \
                             RAD=float(arr[8]), TAX=float(arr[9]), PTRATIO=float(arr[10]), \
                             B=float(arr[11]), LASTAT=float(arr[12]), MEDV=float(arr[13])))
    boston_house_df = sql_context.createDataFrame(row_boston_house_rdd)
    print("原始数据对应的schema信息为:", end='')
    print(boston_house_df.schema)
    boston_house_df.show(truncate=False)

    # 0. 数据划分为训练集合测试集数据
    train_df, test_df = boston_house_df.randomSplit(weights=[0.7, 0.3], seed=28)

    # 1. 特征工程
    # a. 数据合并
    all_feature_name = boston_house_df.schema.names
    all_feature_name.remove('MEDV')
    vector_assembler = VectorAssembler(inputCols=all_feature_name, outputCol='f1')
    train_df_tmp01 = vector_assembler.transform(train_df)
    test_df_tmp01 = vector_assembler.transform(test_df)
    # b. 多项式扩展
    poly = PolynomialExpansion(inputCol='f1', outputCol='f2', degree=3)
    train_df_tmp02 = poly.transform(train_df_tmp01)
    test_df_tmp02 = poly.transform(test_df_tmp01)
    # c. 降维
    pca = PCA(k=75, inputCol='f2', outputCol='features')
    pca_model = pca.fit(train_df_tmp02)
    train_df_tmp03 = pca_model.transform(train_df_tmp02)
    test_df_tmp03 = pca_model.transform(test_df_tmp02)

    # NOTE：因为算法模型的构建是一个迭代的过程，也就是说数据会被多次使用，那么将数据进行缓存操作
    train_df_tmp03.cache()

    # 2. 算法模型构建
    # featuresCol: 给定输入的特征属性，要求是Vector类型，Vector中的数据是double类型
    # labelCol: 需要预测的实际y值，要求是double类型
    # predictionCol：预测值新标签，要求不存在
    # maxIter: 最大迭代次数
    # regParam：学习因子/学习率，当设置为0的时候表示不使用正则项； 计算公式: reg * ((1-p)*L2 + p*L1)
    # elasticNetParam： 给定EN弹性网络中的alpha(p)的值，取值范围为:[0,1],当为0的时候，表示正则项使用L2，为1的时候表示使用L1正则项；计算公式为: (1-p)*L2 + p*L1
    # fitIntercept：是否需要截距，设置为true表示有截距
    # standardization：是否进行标准化操作，默认进行
    linear = LinearRegression(featuresCol="features", labelCol="MEDV", predictionCol="prediction", \
                              maxIter=100, regParam=0.0, elasticNetParam=0.0, \
                              fitIntercept=True, standardization=True)
    # 模型训练
    linear_model = linear.fit(train_df_tmp03)
    # 使用训练好的模型对数据进行转换操作/预测操作
    linear_model.transform(test_df_tmp03).show(truncate=False)

    # 使用GBDT
    gbt = GBTRegressor(featuresCol="features", labelCol="MEDV", predictionCol="prediction", \
                       maxDepth=5, subsamplingRate=0.8, maxIter=20, stepSize=0.1)
    gbt_model = gbt.fit(train_df_tmp03)
    gbt_model.transform(test_df_tmp03).show(truncate=False)

    name_models = [('线性回归', linear_model), ('GBDT', gbt_model)]
    # 3. 模型效果评估
    # metricName: 指定进行何种指标的评估，只支持R^2（r2）， mse, rmse, mae四种
    evaluator = RegressionEvaluator(metricName='r2', labelCol='MEDV', predictionCol='prediction')
    for name, model in name_models:
        print("{}在训练集上R^2={}".format(name, evaluator.evaluate(model.transform(train_df_tmp03))))
        print("{}在测试集上R^2={}".format(name, evaluator.evaluate(model.transform(test_df_tmp03))))
