# -*- coding:utf-8 -*-
import os
import numpy as np
from pyspark import SparkConf, SparkContext

# 给定SPARK_HOME环境变量
if 'SPARK_HOME' not in os.environ:
    os.environ['SPARK_HOME'] = 'D:\syl\dev\spark-1.6.1-bin-2.5.0-cdh5.3.6'

# 1. 创建上下文
# setMaster: 在开发阶段，必须给定，而且只能是：local、local[*]、local[K]；在集群中运行的时候，是通过命令参数给定，代码中最好注释掉
# setAppName：给定应用名称，必须给定
conf = SparkConf() \
    .setMaster('local') \
    .setAppName('demo')
sc = SparkContext(conf=conf)

# 2. RDD构建
rdd1 = sc.parallelize(["spark hadoop hbase spark hbase", "spark hadoop spark hbase", "ml spark hbase dl", "avc de"])

# 3. API应用
"""
map API给定的处理函数f：
接收一个输入参数值（即RDD中的一条数据），返回的值直接就是RDD中的一条新数据
"""
print("-" * 10 + " map " + "-" * 10)
rdd2 = rdd1.map(f=lambda line: line.strip().split(' '))
print("map RDD Value Size:{}; Value:".format(rdd2.count()))
print(rdd2.collect())

"""
flatMap API给定的处理函数f：要求函数f的返回值必须是list、tuple或者数组类型，
接收一个输入参数值（即RDD中的一条数据），返回值为"集合类型", 最终RDD中的元素就是返回集合中的元素
也就是一条数据输入，多条数据输出
"""
print("-" * 10 + " flatMap 1 " + "-" * 10)
rdd3 = rdd1.flatMap(f=lambda line: line.strip().split(' '))
print("flatMap RDD Value Size:{}; Value:".format(rdd3.count()))
print(rdd3.collect())

print("-" * 10 + " flatMap 2 " + "-" * 10)
rdd4 = rdd1.flatMap(f=lambda line: map(lambda word: [word, 1], line.strip().split(' ')))
print("flatMap RDD Value Size:{}; Value:".format(rdd4.count()))
print(rdd4.collect())

"""
filter API给定的处理函数f：要求函数f的返回值必须是boolean类型
接收一个输入参数值（即RDD中的一条数据），返回值为boolean类型
返回为True表示输入的数据值保留，为False表示数据删除。
"""
print("-" * 10 + " filter 1 " + "-" * 10)
rdd5 = rdd1.filter(f=lambda line: len(line) >= 10)
print("filter RDD Value Size:{}; Value:".format(rdd5.count()))
print(rdd5.collect())

print("-" * 10 + " filter 2 " + "-" * 10)
rdd6 = rdd4.filter(lambda word_tuple: word_tuple[0] == 'spark')
print("filter RDD Value Size:{}; Value:".format(rdd6.count()))
print(rdd6.collect())

"""
reduceByKey API: 要求RDD中的数据类型必须是Key/Value的形式
功能：对RDD中的元素按照key进行分组，然后对每一组数据使用给定的函数func进行数据聚合操作
最终API返回的值是：key/聚合value值(函数f)所形成的RDD； 
要求：给定的函数返回的数据类型必须是原始RDD中value的数据类型 
"""
print("-" * 10 + " reduceByKey " + "-" * 10)
rdd7 = rdd4.reduceByKey(func=lambda v1, v2: v1 + v2)
print("reduceByKey RDD Value Size:{}; Value:".format(rdd7.count()))
print(rdd7.collect())

"""
groupByKey API: 要求RDD中的数据类型必须是Key/Value的形式
功能：：对RDD中的元素按照key进行分组，将相当key的value放到一起，形成一个迭代器
最终API返回的值是：key/values值(value形成的迭代器)所形成的RDD；
"""
print("-" * 10 + " groupByKey " + "-" * 10)
rdd8 = rdd4.groupByKey()
print("groupByKey RDD Value Size:{}; Value:".format(rdd8.count()))
rdd8_collect = rdd8.collect()
print(rdd8_collect)
for k, vs in rdd8_collect:
    print("Key={}, Value=(".format(k), end=' ')
    for v in vs:
        print(v, end=", ")
    print(")")

"""
aggregateByKey：要求RDD中的数据类型必须是Key/Value的形式
功能：类似reduceByKey，都是对相同key的数据的value值进行聚合操作，区别是：reduceByKey要求聚合之后的值必须和原始的value值类型一致，aggregateByKey对于聚合之后的值数据类型不做要求。
zeroValue: 对于每个key而言，对应的聚合初始化值
seqFunc: 给定聚合临时聚合值和value值的函数，输入两个参数，函数返回新的聚合值，eg：a就是当前key对应的临时聚合值，b是当前key对应的一个value值
combFunc：给定聚合两个临时聚合值的函数，输入两个参数，函数返回新的聚合值。eg: c和d均为一个临时的聚合值
"""
print("-" * 10 + " aggregateByKey 1 " + "-" * 10)
rdd9 = rdd4.aggregateByKey(zeroValue=0, seqFunc=lambda a, b: a + b, combFunc=lambda c, d: c + d)
print("aggregateByKey RDD Value Size:{}; Value:".format(rdd9.count()))
print(rdd9.collect())

print("-" * 10 + " aggregateByKey 2 " + "-" * 10)
rdd10 = rdd9.map(lambda t: [t[1], t[0]]) \
    .aggregateByKey(zeroValue=np.array([]), seqFunc=lambda a, b: np.append(a, b), combFunc=lambda c, d: np.append(c, d))
print("aggregateByKey RDD Value Size:{}; Value:".format(rdd10.count()))
print(rdd10.collect())
