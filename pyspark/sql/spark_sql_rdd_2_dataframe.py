# -- encoding:utf-8 --
import os
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, Row


def print_iter(iter):
    for v in iter:
        print(v)


# 给定SPARK_HOME环境变量
if 'SPARK_HOME' not in os.environ:
    os.environ['SPARK_HOME'] = 'D:\syl\dev\spark-1.6.1-bin-2.5.0-cdh5.3.6'

# 1. 创建上下文
# setMaster: 在开发阶段，必须给定，而且只能是：local、local[*]、local[K]；在集群中运行的时候，是通过命令参数给定，代码中最好注释掉
# setAppName：给定应用名称，必须给定
conf = SparkConf() \
    .setMaster('local') \
    .setAppName('spark_sql_rdd_2_dataframe')
sc = SparkContext(conf=conf)
sql_context = SQLContext(sparkContext=sc)

# 2. 读取数据形成RDD
dept_rdd = sc.textFile('../datas/dept.txt') \
    .map(lambda line: line.split('\t')) \
    .filter(lambda arr: len(arr) == 3)

# 3. 将RDD转换为DataFrame
# a. 将RDD中的数据类型转换为Row类型
dept_row_rdd = dept_rdd.map(lambda arr: Row(deptno=int(arr[0]), dname=arr[1], loc=arr[2]))
# b. 构建DataFrame
dept_df = sql_context.createDataFrame(dept_row_rdd)

# 4. 查看一下数据及schema信息
print("Schema信息为：")
print(dept_df.schema)
print("数据为:")
dept_df.show()

# 5. DataFrame数据输出
# a. 将DataFrame转换为RDD后对RDD的数据输出
result_rdd = dept_df.rdd.map(lambda row: (row.deptno, row.dname, row.loc))
result_rdd.foreachPartition(lambda iter: print_iter(iter))
# b. 调用DataFrame的相关API直接输出(一般输出到磁盘文件, 文件类型支持：json、orc、parquet)
dept_df.write.mode('overwrite').json('../datas/dept_json')
