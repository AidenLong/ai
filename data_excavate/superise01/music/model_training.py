# -*- coding:utf-8 -*-
import surprise
import pickle
from surprise import KNNBaseline, SVD
from surprise import Dataset, Reader


def train_model1(in_file, out_file):
    """
    基于in_file训练需求一的模型，并保存到out_file
    :param in_file:
    :param out_file:
    :return:
    """
    # 1. 加载用户id、歌曲id、评分的数据
    reader = Reader(line_format='user item rating', sep=',', rating_scale=(1, 10))
    data = Dataset.load_from_file(in_file, reader=reader)

    # 2. 构建训练集
    trainset = data.build_full_trainset()

    # 3. 模型构建
    algo = SVD(n_factors=100, n_epochs=50, reg_all=0.2, lr_all=0.05)

    # 4. 模型训练
    algo.fit(trainset)

    # 5. 模型保存
    surprise.dump.dump(out_file, algo=algo)

    algo.trainset.n_items


def parse_model1_2_recommend_list(model_in_file, out_file, song_id_2_name):
    """
    基于输入的模型model_in_file，构建出每个用户的推荐列表信息
    :param model_in_file:
    :param out_file:
    :param song_id_2_name: 歌曲id和名称之间的映射dict
    :return:
    """
    # 1. 模型加载
    _, algo = surprise.dump.load(model_in_file)

    # 2. 对所有用户产生推荐列表，并将推荐结果保存到目的源(磁盘或者数据库)
    """
    针对每个用户分别产生30个推荐歌曲 ==> 也就是获取评分最高的30首歌作为推荐结果
    """
    with open(out_file, 'w', encoding='utf-8') as writer:
        # 获取用户数量
        n_users = algo.trainset.n_users
        # 获取物品数目
        n_items = algo.trainset.n_items
        print("总用户数目:{}".format(n_users))

        # 遍历所有用户
        for inner_user_id in range(n_users):
            # 用户id转换为外部用户id
            raw_user_id = algo.trainset.to_raw_uid(inner_user_id)
            top_30_rating = []

            # 遍历所有的物品
            for inner_item_id in range(n_items):
                # 物品id转换为外部物品id
                raw_item_id = algo.trainset.to_raw_iid(inner_item_id)

                # 获取得到当前用户对于当前物品的评分
                rating = algo.predict(raw_user_id, raw_item_id).est

                # 根据评分判断是否是最大的30个中的一个
                if len(top_30_rating) < 30:
                    # 如果集合中元素数量都小于30，那么表示推荐的数据还没有推荐30个歌曲，这个时候不考虑评分
                    top_30_rating.append((raw_item_id, rating))
                else:
                    # 当集合中的元素数目大于等于30的时候，那么表示已经获取了30个推荐歌曲，那么此时就需要考虑rating是否比集合中的元素值大，如果大，那么取代集合中评分最小的上商品
                    # a. 集合中的元素按照评分从低到高排序
                    top_30_rating.sort(key=lambda t: t[1])
                    # b. 获取推荐列表中的最小评分
                    min_rating = top_30_rating[0][1]
                    # c. 根据当前评分和最小评分判断是否取代商品
                    if rating > min_rating:
                        # 替换
                        top_30_rating[0] = (raw_item_id, rating)

            # 组合推荐列表信息
            recommend_song = raw_user_id
            for item in top_30_rating:
                item_id = item[0]
                item_name = song_id_2_name[item_id]
                item_rating = item[1]
                recommend_song += "\t" + item_id + "::::" + item_name + "::::" + str(item_rating)

            # 输出信息
            writer.writelines(recommend_song)
            writer.writelines('\n')


def train_model2(in_file, out_file):
    """
    基于in_file训练需求二的模型，并保存到out_file
    :param in_file:
    :param out_file:
    :return:
    """
    # 1. 加载歌单id、歌曲id、评分的数据
    reader = Reader(line_format='user item rating', sep=',', rating_scale=(1, 1))
    data = Dataset.load_from_file(in_file, reader=reader)

    # 2. 构建训练集
    trainset = data.build_full_trainset()

    # 3. 模型构建
    sim_options = {
        'name': 'jaccard',  # 因为我们现在的评分只有1分，所以这里使用jaccard比较适合
        'user_based': True
    }
    algo = KNNBaseline(k=10, sim_options=sim_options)

    # 4. 模型训练
    algo.fit(trainset)

    # 5. 模型保存
    surprise.dump.dump(out_file, algo=algo)


def parse_model2_2_recommend_list(model_in_file, out_file, playlist_id_2_name):
    """
    基于输入的模型model_in_file，构建出每个用户的推荐列表信息
    :param model_in_file:
    :param out_file:
    :param playlist_id_2_name: 歌单id和名称之间的映射dict
    :return:
    """
    # 1. 模型加载
    _, algo = surprise.dump.load(model_in_file)

    # 2. 对所有歌单产生推荐列表，并将推荐结果保存到目的源(磁盘或者数据库)
    """
    当用户在歌单页面的时候，给用户推荐相似的歌单
    ---> 基于歌单的相似度来产生推荐列表 --> 越相似的越推荐
    """
    # 获取歌单数量
    n_playlist = algo.trainset.n_users
    print("总歌单数目:{}".format(n_playlist))

    # 遍历所有歌单
    playlist_id_2_playlists = {}
    for inner_playlist_id in range(n_playlist):
        # 歌单id转换为外部歌单id
        raw_playlist_id = algo.trainset.to_raw_uid(inner_playlist_id)
        top_7_sim = []

        # 遍历所有的歌单
        for inner_playlist_id2 in range(n_playlist):
            # 歌单id转换为外部歌单id
            raw_playlist_id2 = algo.trainset.to_raw_uid(inner_playlist_id2)

            if raw_playlist_id != raw_playlist_id2:
                # 计算歌单之间的相似度
                sim = algo.sim[inner_playlist_id, inner_playlist_id2]

                # 根据相似度判断是否是最大的7个中的一个
                if len(top_7_sim) < 7:
                    # 如果集合中元素数量都小于7，那么表示推荐的数据还没有推荐7个歌单，这个时候不考虑相似度
                    top_7_sim.append((raw_playlist_id2, sim))
                else:
                    # 当集合中的元素数目大于等于7的时候，那么表示已经获取了7个推荐歌单，那么此时就需要考虑sim是否比集合中的元素值大，如果大，那么取代集合中相似度最小的上推荐歌单
                    # a. 集合中的元素按照评分从高到低排序
                    top_7_sim.sort(key=lambda t: t[1], reverse=True)
                    # b. 获取推荐列表中的最小相似度
                    min_sim = top_7_sim[-1][1]
                    # c. 根据当前评分和最小评分判断是否取代商品
                    if sim > min_sim:
                        # 替换
                        top_7_sim[-1] = (raw_playlist_id2, sim)

        # 组合推荐列表信息
        recommend_playlist = "\t".join(
            map(lambda t: t[0] + "::" + playlist_id_2_name[t[0]] + "::" + str(t[1]), top_7_sim))
        playlist_id_2_playlists[raw_playlist_id] = recommend_playlist

    # 推荐结果输出保存
    with open(out_file, 'wb') as writer:
        pickle.dump(playlist_id_2_playlists, writer)
    print("Done!!!!")


if __name__ == '__main__':
    train_flag = False
    music_user_song_rating_file_path = 'D:/syl/ai/data/163_music_user_song_rating.txt'
    music_playlist_song_rating_file_path = 'D:/syl/ai/data/163_music_palylist_song_rating.txt'
    model1_file_path = 'D:/syl/ai/data/model/recommendation1.model'
    model2_file_path = 'D:/syl/ai/data/model/recommendation2.model'
    playlist_pkl_file_path = 'D:/syl/ai/data/playlist.txt'
    song_pkl_file_path = 'D:/syl/ai/data/song.txt'
    model1_result_file_path = 'D:/syl/ai/data/model/recommendation1_result.data'
    model2_result_file_path = 'D:/syl/ai/data/model/recommendation2_result.data'

    if train_flag:
        # 1. 开始模型训练
        print("开始训练模型1......")
        train_model1(music_user_song_rating_file_path, model1_file_path)
        print("模型1训练完成，开始训练模型2......")
        train_model2(music_playlist_song_rating_file_path, model2_file_path)
        print("模型2训练完成!!!!")
    else:
        # 1. 加载id和name之间的映射关系
        song_id_2_name = pickle.load(open(song_pkl_file_path, 'rb'))
        playlist_id_2_name = pickle.load(open(playlist_pkl_file_path, 'rb'))
        # 2. 模型1结果输出
        parse_model1_2_recommend_list(model1_file_path, model1_result_file_path, song_id_2_name)
        # 3. 模型2结果输出
        parse_model2_2_recommend_list(model2_file_path, model2_result_file_path, playlist_id_2_name)
