# -*- coding:utf-8 -*-
import json
import pickle
import random
import time
import numpy as np


def random_sample_playlist(in_file, out_file, max_sample, seed=None):
    """
        歌单数据抽样，输入的数据中一行一个歌单，输出也是一行一个歌单
        :param in_file:  输入文件路径
        :param out_file:  输出文件路径
        :param max_sample: 最多抽取的歌单数目
        :param seed: 随机数种子
        :return:
    """
    if seed:
        random.seed(seed)

    with open(out_file, 'w', encoding='utf-8') as writer:
        with open(in_file, 'r', encoding='utf-8') as reader:
            count = 0
            for line in reader:
                if random.random() < 0.8:
                    writer.writelines(line)
                    count += 1
                    if count % 100 == 0:
                        print(count)

                if count >= max_sample:
                    break

            print('实际抽取的总个数为:%d' % count)


def parse_playlist_2_song(in_playlist):
    """
        将输入的歌单数据转换为我们需要的解析数据，歌单数据格式为json的格式
        :param in_playlist:
        :return:
    """
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
        return False

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
        except Exception as e:
            # print(e)
            pass

    # 返回最终结果
    return str(user_id) + "##" + str(playlist_id) + "##" + playlist_name + \
           "##" + str(playlist_update_time) + "##" + str(playlist_subscribed_count) + \
           "##" + str(playlist_play_count) + song_info + "\n"


def parse_playlist_file(in_file, out_file):
    """
        歌单数据的预处理，将数据进行格式的解析，并输出到目标文件中
        :param in_file:
        :param out_file:
        :return:
    """
    with open(out_file, 'w', encoding='utf-8') as writer:
        with open(in_file, 'r', encoding='utf-8') as reader:
            for line in reader:
                result = parse_playlist_2_song(line)
                if result:
                    writer.writelines(result)


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
        return ','.join([user_id, song_id, str(rating)])
    except Exception as e:
        print(song_info)
        return ''


def parse_user_song_rating_file(in_file, out_file):
    """
    对输入的歌单数据文件进行评分信息的构建，构建结果保存到out_file中
    :param in_file:
    :param out_file:
    :return:
    """
    with open(out_file, 'w', encoding='utf-8') as writer:
        with open(in_file, 'r', encoding='utf-8') as reader:
            for line in reader:
                # 获取歌单信息
                contents = line.strip().split('\t')
                user_id, _, _, update_time, subscribed_count, play_count = contents[0].split("##")
                update_time = float(update_time)
                subscribed_count = float(subscribed_count)
                play_count = float(play_count)

                # 获取歌曲信息
                user_song_info = map(
                    lambda song: parse_user_song_rating(user_id, update_time, subscribed_count, play_count, song),
                    contents[1:])
                user_song_info = filter(lambda t: len(t.split(",")) > 2, user_song_info)

                # 获取输出数据
                result = "\n".join(user_song_info)
                if result:
                    writer.writelines(result)
                    writer.writelines('\n')


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
        return ','.join([playlist_id, song_id, str(rating)])
    except Exception as e:
        # print(song_info)
        return ''


def parse_playlist_song_rating_file(in_file, out_file):
    """
    构建歌单-歌曲评分矩阵，只要歌曲在歌单中出现了，就认为评分为1
    :param in_file: 输入的是经过特征提取之后数据文件路径
    :param out_file:  输出文件路径
    :return:
    """
    with open(out_file, 'w', encoding='utf-8') as writer:
        with open(in_file, 'r', encoding='utf-8') as reader:
            for line in reader:
                # 获取歌单信息
                contents = line.strip().split('\t')
                _, playlist_id, _, _, _, _ = contents[0].split("##")

                # 获取歌单的信息
                playlist_song_info = map(lambda song: parse_playlist_song_rating(playlist_id, song), contents[1:])
                playlist_song_info = filter(lambda t: len(t.split(",")) > 2, playlist_song_info)

                # 获取数据结果
                result = "\n".join(playlist_song_info)
                if result:
                    writer.writelines(result)
                    writer.writelines('\n')


def parse_playlist_song_id_2_name(in_file, out_playlist_file, out_song_file):
    """
    从数据中提取歌单id和歌单名称之间的映射以及歌曲id和歌曲名称之间的映射<br/>
    因为推荐算法中应用的是歌单的id或者歌曲的id，产生推荐结果的时候不直观，所以我们这里的映射主要的目的就是为了后面产生推荐结果的时候看上去比较自然
    :param in_file:  输入的原始数据(解析后的)文件路径
    :param out_playlist_file: 输出歌单id-歌单名称之间的文件路径
    :param out_song_file:  输出歌曲id-歌曲名称之间的文件路径
    :return:
    """
    # 歌单id和歌单名称之间的映射字典
    playlist_id_2_name_dict = {"abc": "dfw", "abd": "123"}
    # 歌曲id和歌曲名称之间的映射字典
    song_id_2_name_dict = {}

    # 从输入数据中获取映射关系
    with open(in_file, 'r', encoding='utf-8') as reader:
        for line in reader:
            try:
                # 获取歌单信息
                contents = line.strip().split('\t')
                _, playlist_id, playlist_name, _, _, _ = contents[0].split("##")
                playlist_id_2_name_dict[playlist_id] = playlist_name

                # 获取歌曲信息
                for song in contents[1:]:
                    try:
                        song_id, song_name, _ = song.split('::::')
                        song_id_2_name_dict[song_id] = song_name
                    except:
                        print("Song format error:", end=song + '\n')
            except:
                print("Playlist format error:", end=line + '\n')

    # 进行数据输出（输出流必须是2进值的形式）
    with open(out_playlist_file, 'wb') as playlist_writer:
        pickle.dump(playlist_id_2_name_dict, playlist_writer)
    with open(out_song_file, 'wb') as song_writer:
        pickle.dump(song_id_2_name_dict, song_writer)


if __name__ == '__main__':
    all_song_file_path = 'D:/syl/ai/data/playlist_detail_music_500.json'
    sample_song_file_path = 'D:/syl/ai/data/playlist_detail_music_500_1.json'
    music_playlist_song_file_path = 'D:/syl/ai/data/163_music_playlist.txt'
    music_user_song_rating_file_path = 'D:/syl/ai/data/163_music_user_song_rating.txt'
    music_playlist_song_rating_file_path = 'D:/syl/ai/data/163_music_palylist_song_rating.txt'
    playlist_pkl_file_path = 'D:/syl/ai/data/playlist.txt'
    song_pkl_file_path = 'D:/syl/ai/data/song.txt'

    # 1. 样本数据抽取
    # random_sample_playlist(all_song_file_path, sample_song_file_path, max_sample=1)

    # 2. 特征属性的提取
    # parse_playlist_file(all_song_file_path, music_playlist_song_file_path)

    # 3. 用户-歌曲评分矩阵的构建
    # parse_user_song_rating_file(music_playlist_song_file_path, music_user_song_rating_file_path)

    # 4. 歌单-歌曲的评分矩阵构建
    # parse_playlist_song_rating_file(music_playlist_song_file_path, music_playlist_song_rating_file_path)

    # 5. id->name数据提取
    parse_playlist_song_id_2_name(music_playlist_song_file_path, playlist_pkl_file_path, song_pkl_file_path)
