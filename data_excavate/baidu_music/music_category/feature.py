# coding:utf-8
import pandas as pd
import numpy as np
import glob
from pydub.audio_segment import AudioSegment
from scipy.io import wavfile
from python_speech_features import mfcc
import os

music_info_csv_file_path = '../data/music_info.csv'
music_audio_dir = '../data/music_category/*.mp3'
music_index_label_path = '../data/music_index_label.csv'
music_features_file_path = '../data/music_features.csv'


def extract_label():
    data = pd.read_csv(music_info_csv_file_path)
    data = data[['name', 'tag']]
    return data


def fetch_index_label():
    """
    从文件中读取index和label之间的映射关系，并返回dict
    """
    data = pd.read_csv(music_index_label_path, header=None, encoding='utf-8')
    name_label_list = np.array(data).tolist()
    index_label_dict = dict(map(lambda t: (t[1], t[0]), name_label_list))
    return index_label_dict


def extract(file):
    items = file.split('.')
    file_format = items[-1].lower()
    file_name = file[: -(len(file_format) + 1)]
    if file_format != 'wav':
        song = AudioSegment.from_file(file, format='mp3')
        file = file_name + '.wav'
        song.export(file, format='wav')
    try:
        rate, data = wavfile.read(file)
        mfcc_feas = mfcc(data, rate, numcep=13, nfft=2048)
        mm = np.transpose(mfcc_feas)
        mf = np.mean(mm, axis=1)  # mf变成104维的向量
        mc = np.cov(mm)
        result = mf
        for i in range(mm.shape[0]):
            result = np.append(result, np.diag(mc, i))
        os.remove(file)
        return result
    except Exception as msg:
        print(msg)


def extract_and_export():  # 主函数
    df = extract_label()
    name_label_list = np.array(df).tolist()
    name_label_dict = dict(map(lambda t: (t[0], t[1]), name_label_list))
    labels = set(name_label_dict.values())
    label_index_dict = dict(zip(labels, np.arange(len(labels))))

    all_music_files = glob.glob(music_audio_dir)
    all_music_files.sort()

    loop_count = 0
    flag = True

    all_mfcc = np.array([])
    for file_name in all_music_files:
        print('开始处理：' + file_name.replace('\xa0', ''))
        music_name = file_name.split('\\')[-1].split('.')[-2].split('-')[-1]
        music_name = music_name.strip()
        if music_name in name_label_dict:
            label_index = label_index_dict[name_label_dict[music_name]]
            ff = extract(file_name)
            ff = np.append(ff, label_index)

            if flag:
                all_mfcc = ff
                flag = False
            else:
                all_mfcc = np.vstack([all_mfcc, ff])
        else:
            print('无法处理：' + file_name.replace('\xa0', '') + '; 原因是：找不到对应的label')
        print('looping-----%d' % loop_count)
        print('all_mfcc.shape:', end='')
        print(all_mfcc.shape)
        loop_count += 1
    # 保存数据
    label_index_list = []
    for k in label_index_dict:
        label_index_list.append([k, label_index_dict[k]])
    pd.DataFrame(label_index_list).to_csv(music_index_label_path, header=None, \
                                          index=False, encoding='utf-8')
    pd.DataFrame(all_mfcc).to_csv(music_features_file_path, header=None, \
                                  index=False, encoding='utf-8')
    return all_mfcc
