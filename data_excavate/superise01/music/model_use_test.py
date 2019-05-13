# -- encoding:utf-8 --
import pickle

if __name__ == '__main__':
    """
    这里的代码模拟的就是线上的部分内容
    当前用户269528599浏览了619752169歌单
    """
    # 1. 加载数据
    model1_result_file_path = 'D:/syl/ai/data/model/recommendation1_result.data'
    model2_result_file_path = 'D:/syl/ai/data/model/recommendation2_result.data'

    user_id_2_recommend_songs = {}
    with open(model1_result_file_path, 'r', encoding='utf-8') as reader:
        for line in reader:
            contents = line.split("\t")
            user_id = contents[0]
            recommend_songs = []
            for song in contents[1:]:
                id, name, _ = song.split("::::")
                recommend_songs.append((id, name))
            user_id_2_recommend_songs[user_id] = recommend_songs

    playlist_id_2_recommend_playlists = pickle.load(open(model2_result_file_path, 'rb'))

    # 当用户进入首页的时候，产生30首歌曲的推荐
    current_user_id = "269528599"
    print("当前用户%s的30首歌曲推荐结果为：" % current_user_id)
    for r_song in user_id_2_recommend_songs[current_user_id]:
        print("\t{}---{}".format(r_song[0], r_song[1]))

    # 当用户浏览某个歌单的时候，产生对应的歌单信息
    current_playlist_id = '619752169'
    print("当前歌单%s最相似的推荐歌单为：" % current_playlist_id)
    recommend_playlist = playlist_id_2_recommend_playlists[current_playlist_id]
    for r_playlist in recommend_playlist.split('\t'):
        r = r_playlist.split('::')
        print("\t{}--{}--{}".format(r[0], r[1], r[2]))
