# -- encoding:utf-8 --
import os
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, Row
import time


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
    .setAppName('sparksqlreaderdataframe')
sc = SparkContext(conf=conf)
sql_context = SQLContext(sparkContext=sc)

# 直接使用SQLContext的API读取json格式的数据形成DataFrame
# df = sql_context.read.json('../datas/dept_json')
df = sql_context.read.json('../datas/dept.json')
df.show()

# 程序暂停一段时间，方便我们观看4040页面，看一下程序的运行情况
time.sleep(120)
