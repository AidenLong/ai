# -*- coding:utf-8 -*-
import numpy as np
import collections


def read_data(fname):
    with open(fname, 'r') as f:
        content = f.readlines()
    # 去掉每行的前后空格
    content = [x.strip() for x in content]
    # 得到单词
    content_size = len(content)
    words = [content[i].split() for i in range(content_size)]
    words = np.array(words)
    words = np.reshape(words, [content_size, -1, 1])

    return words


def build_dataset(words):
    count = collections.Counter(words).most_common()
    # 构建一个字典
    dicts = {}
    k = 0
    for word, _ in count:
        dicts[word] = k
        k += 1
    recerse_dict = dict(zip(dicts.values(), dicts.keys()))

    return dicts, recerse_dict


# 1. 加载数据
training_file = 'belling_the_cat.txt'
training_data = read_data(training_file)
print(training_data)

# 2. 构建单词和数字之间的映射关系
dicts, reverse_dict = build_dataset(words=training_data)
vocab_size = len(dicts)
