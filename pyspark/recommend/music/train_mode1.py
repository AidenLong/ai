# -- encoding:utf-8 --
"""
需求一的pyspark实现：为每个用户产生一个30首歌的推荐列表 => 基于用户-歌曲评分矩阵，使用传统的推荐算法即可完成推荐
"""

import os
import numpy as np
import time
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, Row
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

if 'SPARK_HOME' not in os.environ:
    os.environ['SPARK_HOME'] = 'D:\syl\dev\spark-1.6.1-bin-2.5.0-cdh5.3.6'

music_user_song_rating_file_path = 'file:///D:/syl/ai/data/163_music_user_song_rating'
song_pkl_file_path = 'file:///D:/syl/ai/data/song_pkl'
model_save_file_path = 'file:///D:/syl/ai/data/model1'
model_result_save_file_path = 'file:///D:/syl/ai/data/model1_result'
train_model = False

# 1. 构建上下文
conf = SparkConf() \
    .setMaster('local[10]') \
    .setAppName('train model01')
sc = SparkContext(conf=conf)
sql_context = SQLContext(sparkContext=sc)

# 2. 数据读取
df = sql_context.read.json(music_user_song_rating_file_path)

if train_model:
    # 3. 将数据转换为Rating形式的RDD
    rating_rdd = df.rdd.map(lambda row: Rating(int(row.user_id), int(row.item_id), float(row.rating)))
    # 4. 数据缓存
    rating_rdd.cache()
    # 5. 模型训练
    # rank：隐因子的数目
    rank = 10
    # 迭代次数
    num_iterations = 10
    model = ALS.train(rating_rdd, rank, num_iterations)
    # 6. 模型保存
    model.save(sc, model_save_file_path)
else:
    # 3. 提取数据并缓存
    user_item_rating_rdd = df.rdd.map(lambda row: (int(row.user_id), int(row.item_id), float(row.rating)))
    user_item_rating_rdd.cache()

    # 4. 加载模型
    same_model = MatrixFactorizationModel.load(sc, model_save_file_path)

    # 5. 对所有用户、所有物品做一个预测的操作（构建过程）
    all_user_id_rdd = user_item_rating_rdd.map(lambda arr: arr[0]).distinct(1)
    all_item_id_rdd = user_item_rating_rdd.map(lambda arr: arr[1]).distinct(5)
    all_user_id_rdd.cache()
    all_item_id_rdd.cache()
    all_user_2_item_id_rdd = all_user_id_rdd.cartesian(all_item_id_rdd)
    print("用户数目:{}".format(all_user_id_rdd.count()))
    print("物品数目:{}".format(all_item_id_rdd.count()))

    # 6. 对所有用户、所有物品做一个预测的操作（预测过程）
    predict_rating = same_model.predictAll(all_user_2_item_id_rdd) \
        .map(lambda r: ((r[0], r[1]), float(np.clip(r[2], 1.0, 10.0))))
    predict_rating.cache()

    # 7. 计算MSE指标
    rates_rating = user_item_rating_rdd.map(lambda arr: ((arr[0], arr[1]), arr[2]))
    rates_rating.cache()
    rates_and_predict_rating = rates_rating.join(predict_rating)
    mse = rates_and_predict_rating.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean()
    print("Mean Squared Error = " + str(mse))


    # 8. 结果输出(假定要求输出的推荐列表不能包含用户已评估过的数据)
    def seq_func(arr, b):
        if len(arr) < 30:
            arr.append(b)
        else:
            sorted_arr = sorted(arr, key=lambda t: t[1])
            min_item = sorted_arr[0]
            if b[1] > min_item[1]:
                sorted_arr[0] = b
            arr = sorted_arr
        return arr


    def comb_func(arr1, arr2):
        for v in arr2:
            arr1 = seq_func(arr1, v)
        return arr1


    def merge_recommendation(t):
        user_id = t[0]
        recommendation_info = ''
        for item_id, rating in t[1]:
            recommendation_info += '\t' + str(item_id) + '::::' + str(rating)
        return str(user_id) + recommendation_info


    predict_rating.leftOuterJoin(rates_rating) \
        .filter(lambda t: t[1][1] is None) \
        .map(lambda t: (t[0][0], (t[0][1], t[1][0]))) \
        .aggregateByKey(zeroValue=[], seqFunc=lambda a, b: seq_func(a, b), combFunc=lambda c, d: comb_func(c, d)) \
        .map(lambda t: merge_recommendation(t)) \
        .saveAsTextFile(model_result_save_file_path)

# 为了看一下4040页面，暂停一小会儿
time.sleep(120)
