# -- encoding:utf-8 --
"""
PySpark中实现Fp Tree的案例代码
"""

import os
from pyspark import SparkConf, SparkContext
from pyspark.mllib.fpm import FPGrowth
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

# 2. 构建RDD(RDD就相当于交易集，RDD中的每条记录就相当一个样本数据)
items = [
    ['A', 'B', 'C', 'E', 'F', 'O'],
    ['A', 'C', 'G'],
    ['E', 'I'],
    ['A', 'C', 'D', 'E', 'G'],
    ['A', 'C', 'E', 'G', 'L'],
    ['E', 'J'],
    ['A', 'B', 'C', 'E', 'F', 'P'],
    ['A', 'C', 'D'],
    ['A', 'C', 'E', 'G', 'M'],
    ['A', 'C', 'E', 'G', 'N']
]
rdd = sc.parallelize(items)
print("第一条样本数据:{}".format(rdd.first()))

# 3. 构建模型
# minSupport：给定最小支持度
model = FPGrowth.train(data=rdd, minSupport=0.2)

# 4. 需求一：获取所有的频繁项集形成的RDD
all_freq_itemsets_rdd = model.freqItemsets()
print("所有的频繁项集")
print(all_freq_itemsets_rdd.collect())

# 5. 需求二：获取频繁项集中长度为3的频繁项集
three_freq_itemsets_rdd = all_freq_itemsets_rdd \
    .filter(lambda freqItemsset: len(freqItemsset.items) == 3)
print("长度为3的频繁项集:")
print(three_freq_itemsets_rdd.collect())

# 6. 需求三：获取指定包含某个项(A)的并且长度为3的最大支持度的频繁项集
# a. 获取包含A并且长度为3的频繁项集
include_A_three_freq_itemsets_rdd = three_freq_itemsets_rdd \
    .filter(lambda freqItemsset: 'A' in freqItemsset.items)
# b. 获取支持度最大的频繁项集
max_include_A_three_freq_itemset = include_A_three_freq_itemsets_rdd \
    .max(key=lambda freqItemsset: freqItemsset.freq)
print("最大频繁项集:{}".format(max_include_A_three_freq_itemset))

"""
关联规则应用到推荐系统中：
1. 计算lhs -> rhs(support)，对于每个lhs来讲，保存最大的support对应的rhs
2. 当用户浏览\购买lhs商品的时候，选择对应的rhs推荐给该用户
"""

# 需求四
"""
针对于"'C', 'E', 'A'", 可以有一下三种情况：
1. lhs=C, rhs=A,E, support=6
2. lhs=E, rhs=A,C, support=6
3. lhs=A, rhs=C,E, support=6
"""

# 给定数据库连接信息
url = 'jdbc:mysql://localhost:3306/test?serverTimezone=GMT%2B8&useUnicode=true&characterEncoding=utf-8'
table = 'tb_tf_tree'
properties = dict()
properties['user'] = 'root'
properties['password'] = '123456'


def f(freq_itemset):
    """
    将一条输入数据转换为多条输出数据: lhs, rhs, support
    :param freq_itemset:
    :return:
    """
    return map(lambda item: Row(lhs=item, rhs=','.join([i for i in freq_itemset.items if i != item]),
                                support=freq_itemset.freq), freq_itemset.items)


# 获取数据的数据
all_freq_itemsets_row_rdd = all_freq_itemsets_rdd \
    .filter(lambda freq_itemset: len(freq_itemset.items) == 3) \
    .flatMap(lambda freq_itemset: f(freq_itemset)) \
    .map(lambda row: (row.lhs, row)) \
    .reduceByKey(func=lambda row1, row2: row1 if row1.support > row2.support else row2) \
    .map(lambda t: t[1])
# 构建成DataFrame后进行数据输出
sqlContext.createDataFrame(all_freq_itemsets_row_rdd) \
    .write \
    .jdbc(url=url, table=table, mode='overwrite', properties=properties)
print("Done!!!!!")
