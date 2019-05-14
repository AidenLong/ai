# -- encoding:utf-8 --

import os
import numpy as np
import time
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, Row

if 'SPARK_HOME' not in os.environ:
    os.environ['SPARK_HOME'] = 'D:\syl\dev\spark-1.6.1-bin-2.5.0-cdh5.3.6'

all_song_file_path = 'file:///D:/syl/ai/data/playlist_detail_music_500.json'
music_user_song_rating_file_path = 'file:///D:/syl/ai/data/163_music_user_song_rating'
music_playlist_song_rating_file_path = 'file:///D:/syl/ai/data/163_music_palylist_song_rating'
playlist_pkl_file_path = 'file:///D:/syl/ai/data/playlist_pkl'
song_pkl_file_path = 'file:///D:/syl/ai/data/song_pkl'

# 1. 构建上下文
conf = SparkConf() \
    .setMaster('local[10]') \
    .setAppName('data transform02')
sc = SparkContext(conf=conf)
sql_context = SQLContext(sparkContext=sc)

# 2. 读取数据，并查看数据结构
df = sql_context.read.json(all_song_file_path)
# print("Schema信息为：{}".format(df.schema))
# df.show(truncate=False)
# df.select('result.id', 'result.userId').show(truncate=False)

# 3. 数据抽样
# fraction = 1.0 * 500 / df.count()
# df = df.sample(withReplacement=False, fraction=fraction, seed=28)

# 4. 信息提取
df = df.select('result.userId', 'result.id', 'result.name', 'result.updateTime', 'result.subscribedCount',
               'result.playCount', 'result.tracks')
user_playlist_song_rdd = df.rdd \
    .filter(lambda row: row.subscribedCount >= 100 and row.playCount >= 1000) \
    .flatMap(lambda row: map(lambda song: (row.id, row.name, row.userId, row.subscribedCount, row.playCount,
                                           row.updateTime, song.id, song.name, song.popularity), row.tracks)) \
    .repartition(numPartitions=10)
user_playlist_song_rdd.cache()

# print("信息抽取之后的数据的Schema信息:{}".format(df.schema))
# df.show(truncate=False)
# print(user_playlist_song_rdd.collect())

# 5. 用户-歌曲评分矩阵构建
user_song_rating_row_rdd = user_playlist_song_rdd.map(lambda arr: (arr[2], arr[6], arr[8], arr[3], arr[4], arr[5])) \
    .map(lambda arr: (arr[0], arr[1], arr[2],
                      arr[3] > 10000, arr[4] > 1000, int(time.time()) * 1000 - arr[5] < 31536000000)) \
    .map(lambda arr: (arr[0], arr[1], arr[2], arr[3] and arr[4] and arr[5], arr[3] or arr[4] or arr[5])) \
    .map(lambda arr: (arr[0], arr[1], arr[2], 1.1 if arr[3] else (1.0 if arr[4] else 0.9))) \
    .map(lambda arr: (arr[0], arr[1], np.clip(arr[2] * arr[3] / 10.0, 1.0, 10.0))) \
    .map(lambda arr: Row(user_id=arr[0], item_id=arr[1], rating=float(arr[2]))) \
    .distinct(numPartitions=1)

# 6. 歌单-歌曲评分矩阵构建
playlist_song_rating_row_rdd = user_playlist_song_rdd.map(lambda arr: (arr[0], arr[6], 1.0)) \
    .map(lambda arr: Row(user_id=arr[0], item_id=arr[1], rating=arr[2])) \
    .distinct(numPartitions=1)

# 7. id和name的映射关系获取
playlist_id_2_name_row_rdd = user_playlist_song_rdd.map(lambda arr: Row(id=arr[0], name=arr[1])).distinct(1)
song_id_2_name_row_rdd = user_playlist_song_rdd.map(lambda arr: Row(id=arr[6], name=arr[7])).distinct(1)

# 结果数据保存
sql_context.createDataFrame(user_song_rating_row_rdd) \
    .write.mode('overwrite').json(music_user_song_rating_file_path)
sql_context.createDataFrame(playlist_song_rating_row_rdd) \
    .write.mode('overwrite').json(music_playlist_song_rating_file_path)
sql_context.createDataFrame(playlist_id_2_name_row_rdd) \
    .write.mode('overwrite').json(playlist_pkl_file_path)
sql_context.createDataFrame(song_id_2_name_row_rdd) \
    .write.mode('overwrite').json(song_pkl_file_path)

# 为了看一下运行情况，暂停一段时间
time.sleep(120)
