# -*- coding:utf-8 -*-
"""
在本地开发环境运行spark应用程序的时候，必须给定spark的相关内容。即必须给定SPARK_HOME环境变量
给定该环境变量的作用主要是：将spark相关的配置信息、jar文件等内容添加到应用的classpath环境变量中
"""

"""
强调：环境配置的Python版本必须和开发工具中的Python版本一致。(命令行执行以下python -V，看一下版本是否和开发工具中的一致)
"""

import os
import time
from pyspark import SparkConf, SparkContext


def print_iter(iter):
    for v in iter:
        print("单词:%s，出现次数:%d" % v)


if 'SPARK_HOME' not in os.environ:
    os.environ['SPARK_HOME'] = 'D:\syl\dev\spark-1.6.1-bin-2.5.0-cdh5.3.6'

# 1. SparkContext上下文构建
# setMaster: API作用：给定应用程序在哪儿运行；在开发阶段，代码中给定，而且只能给定local；在生成环境中，该API不允许调用，通过命令spark-submit的--master参数给定
# setAppName：设置应用程序的名称
conf = SparkConf() \
    .setMaster('local') \
    .setAppName('wordcount')
sc = SparkContext(conf=conf)

# 2. 读取文件数据，形成RDD
# textFile：从给定的文件路径中读取文本格式数据，形成的RDD中就是文件中的数据，一行数据就是RDD中的一条数据
rdd = sc.textFile('../datas/wc.txt')
# rdd = sc.textFile('hdfs://192.168.40.128:8020/test/input/wc.txt')

# 3. 对数据做一个处理(实现WordCount需求)
"""
RDD API说明：
map: 会返回一个新的RDD，使用给定的函数对RDD中的每一条数据做转换操作；函数的输入就是RDD中的每一条数据，函数的输出就是新RDD中的数据。
flatMap: 会返回一个新的RDD，功能：扁平化操作 + map转换操作；要求给定的函数返回类型是集合/迭代器类型。使用给定的函数对RDD中的每一条数据做转换操作，每条数据转换之后输出值是多个(迭代器、集合等)，函数的输入就是RDD中的每一条数据，函数的输出是迭代器，迭代器中的每个元素是新RDD中的数据
filter: 过滤函数，使用给定的函数对RDD中的每一条数据进行判断，函数输入是RDD的每一条数据，函数输出是True、False；当输出为True的时候，表示该数据保留，也就是在新RDD中存在，如果输出为False，表示数据不保留，也就是在新RDD中不存在
groupByKey: 要求RDD中的数据类型必须是二元组的形式，API作用：将所有相同key的数据的value值合并成为一个迭代器，最终返回的RDD中的元素是一个二元组的类型，二元组的第一个元素是key类型，第二个元素是一个迭代器类型，迭代器中的数据是原RDD中的value数据
foreachPartition: 该API没有返回值。对RDD中的每个分区的数据使用给定的函数进行处理，函数的输入就是一个分区数据形成的迭代器
reduceByKey: 要求RDD中的数据类型必须是二元组的形式，API作用：对相同key的数据的value值使用给定的func函数坐value数据的合并操作；v1表示当前key对应的临时聚合值，v2表示当前key对应的一个value值；最终返回的RDD是一个二元组的RDD；RDD中的第一个元素是key，第二个元素是聚类之后的聚合值
"""
# result_rdd = rdd.flatMap(f=lambda line: line.split(' ')) \
#     .filter(f=lambda word: len(word) != 0) \
#     .map(f=lambda word: (word, 1)) \
#     .groupByKey() \
#     .map(f=lambda t: (t[0], len(t[1])))

result_rdd = rdd.flatMap(f=lambda line: line.split(' ')) \
    .filter(f=lambda word: len(word) != 0) \
    .map(f=lambda word: (word, 1)) \
    .reduceByKey(func=lambda v1, v2: v1 + v2)

# 4. 结果输出
# saveAsTextFile: 将RDD的结果以文本文件的形式存储。要求给定的文件路径不存在；否则报错
# result_rdd.map(f=lambda t: t[0] + ' ' + str(t[1])).saveAsTextFile('../datas/wc_result')
# 直接打印输出
result_rdd.foreachPartition(lambda iter: print_iter(iter))

# 5. 获取出现此时最多的前2个单词
top_2_list = result_rdd.map(f=lambda t: (t[1], t[0])).top(2)
print(top_2_list)

# 暂停一段时间，方便看4040页面
time.sleep(100)

# 6. 关闭sparkcontext
sc.stop()
