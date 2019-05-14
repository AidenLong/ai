# --encoding:utf-8 --

import os
import re
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, Row
from pyspark.ml.feature import PolynomialExpansion, VectorAssembler, PCA
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

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
    # b. 多项式扩展
    poly = PolynomialExpansion(inputCol='f1', outputCol='f2', degree=3)
    # c. 降维
    pca = PCA(k=75, inputCol='f2', outputCol='features')

    # NOTE：因为算法模型的构建是一个迭代的过程，也就是说数据会被多次使用，那么将数据进行缓存操作
    train_df.cache()

    # 2. 算法模型构建
    linear = LinearRegression(featuresCol="features", labelCol="MEDV", predictionCol="prediction")
    # 3. 构建管道对象
    pipline = Pipeline(stages=[vector_assembler, poly, pca, linear])

    # 4. 网格交叉验证
    evaluator = RegressionEvaluator(metricName='r2', labelCol='MEDV', predictionCol='prediction')
    grid = ParamGridBuilder() \
        .addGrid(linear.maxIter, [10, 100]) \
        .addGrid(poly.degree, [2, 3]) \
        .build()
    cv = CrossValidator(estimator=pipline, estimatorParamMaps=grid, evaluator=evaluator)
    # 模型训练
    cv_model = cv.fit(train_df)
    print("最优参数模型：", end='')
    print(cv_model.bestModel.stages[-1])

    # 4. 模型效果评估
    print("网格交叉验证模型在训练集上R^2={}".format(evaluator.evaluate(cv_model.transform(train_df))))
    print("网格交叉验证模型在测试集上R^2={}".format(evaluator.evaluate(cv_model.transform(test_df))))
