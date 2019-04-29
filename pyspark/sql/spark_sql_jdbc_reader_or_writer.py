# -- encoding:utf-8 --
import os
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, Row
import time


def print_iter(iter):
    for v in iter:
        print(v)


url = 'jdbc:mysql://localhost:3306/test?characterEncoding=utf8&serverTimezone=UTC'
table = 'tmp_dept'
properties = dict()
properties['user'] = 'root'
properties['password'] = '123456'

"""
如果需要将数据输出到mysql或者从mysql读取数据的时候，必须启动mysql服务，并且添加msyql的链接驱动jar文件
启动对应位置的mysql服务
使用这种方式的时候，比如把mysql的driver驱动添加到环境中(spark应用环境)
添加方式：
-1. 将mysql-connector-java-5.1.27-bin.jar文件放到spark根目录下的lib文件夹中，即D:\ProgramFiles\spark-1.6.1-bin-2.5.0-cdh5.3.6\lib
-2. 将放置的mysql的jar文件修改为以: datanucleus-的文件名称，eg：datanucleus-mysql-connector-java-5.1.27-bin.jar
"""

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

# 4. 数据输出到关系型数据库(使用两种方式)
# a. 将DataFrame转换为RDD，然后使用RDD的foreachPartition API，在给定的函数中调用相关的数据输出API(mysql)
# def save_to_mysql(iter):
#     """
#     输出数据到mysql数据库
#     :param iter:
#     :return:
#     """
#     # 1. 创建mysql的链接connection
#     # 2. 构建pstmt对象
#     # 3. 遍历数据进行输出
#     for v in iter:
#         deptno = v[0]
#         dname = v[1]
#         loc = v[2]
#         # 这里将数据输出
#     # 4. 关闭链接
#     pass
#
#
# dept_df.rdd.map(lambda row: (row.deptno, row.dname, row.loc)) \
#     .foreachPartition(lambda iter: save_to_mysql(iter))

# b. 直接调用DataFrame的相关API进行数据的输出
# 这里mode指定的意思是，表是否存在的时候采用的策略
dept_df.write.jdbc(url, table, mode='overwrite', properties=properties)

# 5. 从Mysql加载数据形成DataFrame
df = sql_context.read.jdbc(url, table, properties=properties)
df.show()
