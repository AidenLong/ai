# -- encoding:utf-8 --
import os
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, Row
import time

# 给定SPARK_HOME环境变量
if 'SPARK_HOME' not in os.environ:
    os.environ['SPARK_HOME'] = 'D:\syl\dev\spark-1.6.1-bin-2.5.0-cdh5.3.6'

# 1. 创建上下文
# setMaster: 在开发阶段，必须给定，而且只能是：local、local[*]、local[K]；在集群中运行的时候，是通过命令参数给定，代码中最好注释掉
# setAppName：给定应用名称，必须给定
conf = SparkConf() \
    .setMaster('local') \
    .setAppName('sparksqlreaderdataframe')
sc = SparkContext(conf=conf)
sql_context = SQLContext(sparkContext=sc)

# 2. 读取数据形成RDD
dept_rdd = sc.textFile('../datas/dept.txt') \
    .map(lambda line: line.split('\t')) \
    .filter(lambda arr: len(arr) == 3)

# 3. RDD转换成为DataFrame
dept_row_rdd = dept_rdd.map(lambda arr: Row(deptno=int(arr[0]), dname=arr[1], loc=arr[2]))
dept_df = sql_context.createDataFrame(dept_row_rdd)

# 4. 使用DSL语法(即DataFrame上的API直接数据的读取操作)
dept_df.show()
dept_df.select('deptno', 'dname').show()

dept_df.registerTempTable('tmp_dept')
sql_context.sql("select deptno, dname from tmp_dept").show()
