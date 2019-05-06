# -- encoding:utf-8 --

import os
from pyspark import SparkConf, SparkContext
from pyspark.mllib.fpm import PrefixSpan
from pyspark.sql import SQLContext, Row

# 给定环境变量
if 'SPARK_HOME' not in os.environ:
    os.environ['SPARK_HOME'] = 'D:\syl\dev\spark-1.6.1-bin-2.5.0-cdh5.3.6'

# 1. 创建上下文
conf = SparkConf() \
    .setMaster('local[*]') \
    .setAppName('fp tree')
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sparkContext=sc)

# 2. 构建RDD
"""
案例一：PPT上的案例
"""
# sequence = [
#     [['a'], ['a', 'b', 'c'], ['a', 'c'], ['d'], ['c', 'f']],
#     [['a', 'd'], ['c'], ['b', 'c'], ['a', 'e']],
#     [['e', 'f'], ['a', 'b'], ['d', 'f'], ['c'], ['b']],
#     [['e'], ['g'], ['a', 'f'], ['c'], ['b'], ['c']]
# ]
"""
案例二：假设我们公司是一个电商网站，用户在浏览商品的时候，是根据用户自己的需求来浏览的，而如果某两个客户具有相同的需求的时候，他们的浏览轨迹基本是相同的---> 也就是说，我们可以认为：如果两个人的浏览轨迹相同的话，那么可以认为这两个人的需求也是相同的 --> 可以根据当前用户的浏览轨迹，从历史数据中找出最频繁的对应序列轨迹，将这个轨迹中的商品认为是当前用户可能会在接下来的过程中浏览的商品列表，推荐给当前用户
eg：
 1. 当前用户浏览了商品a，那可以将以a开头的频繁序列的后续序列商品推荐给当前用户
"""
sequence = [
    [['a'], ['b'], ['c'], ['d'], ['a'], ['f']],  # 这就是一条用户的访问路径：a -> b -> c -> d -> a -> f
    [['b'], ['d'], ['a'], ['f']],
    [['c'], ['a'], ['b'], ['f'], ['b']],
    [['f'], ['a'], ['f'], ['b']]
]
sequence_rdd = sc.parallelize(sequence)

# 3. 模型训练
# minSupport：给定频繁序列的阈值支持度大小是多少
# maxPatternLength：指定最长的频繁序列的长度是多少
model = PrefixSpan.train(data=sequence_rdd, minSupport=0.5, maxPatternLength=10)

# 4. 需求一：获取所有频繁序列
all_freq_sequences_rdd = model.freqSequences()
print(all_freq_sequences_rdd.collect())


# 5. 需求二：获取所有长度为3的频繁序列(包含3个项的序列)
def flat_seq(seq):
    """
    扁平化处理
    :param seq:
    :return:
    """
    result = []
    for ss in seq:
        for s in ss:
            result.append(s)
    return result


three_freq_sequences_rdd = all_freq_sequences_rdd \
    .filter(lambda freq_sequence: len(flat_seq(freq_sequence.sequence)) == 3)
print(three_freq_sequences_rdd.collect())

# 6. 需求三：获取指定开头为某个项的并且长度为3的最大支持度的频繁序列
"""
1. 获取长度为3的频繁序列
2. 在第一步的基础上获取开头为某个序列的频繁序列
3. 在第二步的基础上，获取freq最大的频繁序列
"""


def start_with_charts(seq, chs):
    flated_seq = flat_seq(seq)
    for index, ch in enumerate(chs):
        if ch != flated_seq[index]:
            return False
    return True


max_freq_sequence = three_freq_sequences_rdd \
    .filter(lambda freq_sequence: start_with_charts(freq_sequence.sequence, ['a', 'f'])) \
    .max(lambda freq_sequence: freq_sequence.freq)
print(max_freq_sequence)
