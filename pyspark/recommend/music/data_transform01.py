# -- encoding:utf-8 --

import os
import json
import numpy as np
import time
from pyspark import SparkConf, SparkContext, StorageLevel

if 'SPARK_HOME' not in os.environ:
    os.environ['SPARK_HOME'] = 'D:\syl\dev\spark-1.6.1-bin-2.5.0-cdh5.3.6'

all_song_file_path = 'file:///D:/syl/ai/data/playlist_detail_music_500.json'
music_user_song_rating_file_path = 'file:///D:/syl/ai/data/163_music_user_song_rating.txt'
music_playlist_song_rating_file_path = 'file:///D:/syl/ai/data/163_music_palylist_song_rating.txt'
playlist_pkl_file_path = 'file:///D:/syl/ai/data/playlist.txt'
song_pkl_file_path = 'file:///D:/syl/ai/data/song.txt'

# 1. 构建上下文
conf = SparkConf() \
    .setMaster('local[2]') \
    .setAppName('data transform01')
sc = SparkContext(conf=conf)

# 2.读取数据形成RDD
rdd = sc.textFile(all_song_file_path)

# 因为rdd多次使用，所以进行缓存，考虑一下你的运行环境(这里有可能有问题，因为rdd的数据太大，本地内存太小)
# rdd.persist(storageLevel=StorageLevel.MEMORY_AND_DISK)

# 3. 获取总样本数
total_playlist = rdd.count()
# total_playlist = 35320

# 4. 数据抽样
# sample_playlist_number = 500
# sample_rdd = rdd.sample(withReplacement=False, fraction=1.0 * sample_playlist_number / total_playlist)

sample_rdd = rdd


# 5. 数据提取操作
def parse_playlist_2_song(in_playlist):
    """
    将输入的歌单数据转换为我们需要的解析数据，歌单数据格式为json的格式
    :param in_playlist:
    :return:
    """
    try:
        # 歌单数据的转换
        data = json.loads(in_playlist)

        # 开始获取我们的数据
        result_data = data['result']
        # 1. 获取歌单数据
        user_id = result_data['userId']
        playlist_id = result_data['id']
        playlist_name = result_data['name']
        playlist_subscribed_count = result_data['subscribedCount']
        playlist_play_count = result_data['playCount']
        playlist_update_time = result_data['updateTime']

        # 2. 因为存在无用歌单：认为如果歌单的订阅数小于100或者播放数小于1000的时候，认为该歌单没有什么价值，直接过滤掉
        if playlist_play_count < 1000 or playlist_subscribed_count < 100:
            return []

        # 3. 获取歌曲的信息
        song_info = ''
        songs = result_data['tracks']
        for song in songs:
            try:
                # 获取歌曲信息
                song_id = song['id']
                song_name = song['name']
                song_popularity = song['popularity']
                song_info += '\t' + '::::'.join([str(song_id), song_name, str(song_popularity)])
            except:
                pass

        # 返回最终结果
        return [user_id, playlist_id, playlist_name, playlist_update_time, playlist_subscribed_count,
                playlist_play_count, song_info.strip()]
    except:
        return []


feature_rdd = sample_rdd.map(lambda playlist: parse_playlist_2_song(playlist)).filter(lambda arr: len(arr) == 7)

feature_rdd.cache()


# 6. 用户-歌曲评分信息构建


def clip(x, lower_bound, upper_bound):
    """
    当x的值小于lower_bound的时候，返回lower_bound；当x的值大于upper_bound是时候，返回upper_bound
    :param x:
    :param lower_bound:
    :param upper_bound:
    :return:
    """
    return np.clip(x, lower_bound, upper_bound)


def is_last_time(update_time):
    return int(time.time()) * 1000 - update_time < 31536000000


def parse_user_song_rating(user_id, playlist_update_time, playlist_subscribed_count, playlist_play_count, song_info):
    """
    将歌单数据和歌曲数据转为MovieLens格式的数据：用户id、歌曲id、评分
    :param user_id:  用户id
    :param playlist_update_time: 歌单的最新更新数据
    :param playlist_subscribed_count: 歌单的订阅数
    :param playlist_play_count: 歌单的播放数目
    :param song_info:  歌曲信息
    :return:
    """
    try:
        song_id, _, song_popularity = song_info.split('::::')

        # 计算用户对于歌曲的评分
        # 计算规则：使用歌曲的热度作为评分值，而且如果评阅次数超过1000次，并且播放次数超过1w次，同时最新更新时间在一年以内的，评分增加1.1的权重；如果这三个条件，只满足其中的任意一个，权重为1；如果都不满足，那么权重为0.9
        w = 1.0
        if playlist_play_count > 1000 and playlist_subscribed_count > 10000 and \
                is_last_time(playlist_update_time):
            w = 1.1
        elif playlist_play_count <= 1000 and playlist_subscribed_count <= 10000 and \
                (not is_last_time(playlist_update_time)):
            w = 0.9
        rating = float(song_popularity) * w
        # 将rating的取值范围变为[1,10]之间
        rating = clip(rating / 10.0, 1.0, 10.0)

        # 返回结果
        return [user_id, song_id, float(rating)]
    except Exception as e:
        return []


user_song_rating_rdd = feature_rdd.map(lambda arr: [arr[0], arr[3], arr[4], arr[5], arr[6]]) \
    .flatMap(lambda arr: map(lambda song_info: parse_user_song_rating(arr[0], arr[1], arr[2], arr[3], song_info),
                             arr[4].split('\t'))) \
    .filter(lambda arr: len(arr) == 3) \
    .map(lambda arr: (arr[0], arr[1], arr[2])) \
    .distinct(numPartitions=1)


# 7. 歌单-歌曲信息构建

def parse_playlist_song_rating(playlist_id, song_info):
    """
    将歌单数据和歌曲数据转换为MovieLens的格式：歌单id、歌曲id、评分
    :param playlist_id:
    :param song_info:
    :return:
    """
    try:
        song_id, _, _ = song_info.split('::::')

        # 计算歌单对于歌曲的评分
        # 计算规则：只要歌曲在歌单中，评分为1
        rating = 1.0

        # 返回结果
        return [playlist_id, song_id, rating]
    except Exception as e:
        # print(song_info)
        return []


playlist_song_rating_rdd = feature_rdd.map(lambda arr: [arr[1], arr[6]]) \
    .flatMap(lambda arr: map(lambda song_info: parse_playlist_song_rating(arr[0], song_info), arr[1].split('\t'))) \
    .filter(lambda arr: len(arr) == 3) \
    .map(lambda arr: (arr[0], arr[1], arr[2])) \
    .distinct(numPartitions=1)

# id和name的映射
playlist_id_2_name = feature_rdd.map(lambda arr: (arr[1], arr[2])).distinct(numPartitions=1)
song_id_2_name = feature_rdd.map(lambda arr: arr[6]) \
    .flatMap(lambda songs: map(lambda song: song.split('::::'), songs.split('\t'))) \
    .filter(lambda arr: len(arr) >= 2) \
    .map(lambda arr: (arr[0], arr[1])) \
    .distinct(numPartitions=1)

# 结果数据输出
user_song_rating_rdd.saveAsTextFile(music_user_song_rating_file_path)
playlist_song_rating_rdd.saveAsTextFile(music_playlist_song_rating_file_path)
playlist_id_2_name.saveAsTextFile(playlist_pkl_file_path)
song_id_2_name.saveAsTextFile(song_pkl_file_path)

# 为了看一下4040页面，暂停一下
time.sleep(1200)
