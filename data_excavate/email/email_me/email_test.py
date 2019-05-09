# -*- coding:utf-8 -*-
import os
import pandas as pd
import re

# 文件名，标签获取
step1_path = '../data/full/index'
type_dict = {'spam': '1', 'ham': '0'}
index_dict = {}
index_file = open(step1_path)
for step1 in index_file:
    arr = step1.split(' ')
    key, value = arr
    # print(key, value)
    value = value.replace('\n', '')
    index_dict[value] = type_dict[key]
index_file.close()
# print(index_dict)

# 右键文件处理，将标签和结果保存到文件中
with open('../process', mode='w', encoding='utf-8') as write_file:
    for path, label in index_dict.items():
        with open(path, mode='r', encoding='gb2312', errors='ignore') as file:
            content_dict = {}
            is_content = False
            for line in file:
                line = line.strip()
                if not is_content:
                    if line.startswith('From:'):
                        content_dict['from'] = line[5:]
                    elif line.startswith('To:'):
                        content_dict['to'] = line[3:]
                    elif line.startswith('Date:'):
                        content_dict['date'] = line[5:]
                    elif not line:
                        is_content = True
                else:
                    if 'content' in content_dict:
                        content_dict['content'] += line
                    else:
                        content_dict['content'] = line
        print(content_dict)
        content_str = content_dict.get('from', 'unkown').replace(',', '').strip() + ','
        content_str += content_dict.get('to', 'unkown').replace(',', '').strip() + ','
        content_str += content_dict.get('date', 'unkown').replace(',', '').strip() + ','
        content_str += content_dict.get('content', 'unkown').replace(',', '').strip() + ','
        content_str += label + '\n'
        write_file.writelines(content_str)

df = pd.read_csv('../process', sep=',', header=None, names=['from', 'to', 'date', 'content', 'label'])
print(df)


def email_re(str1):
    it = re.findall(r'@([A-Za-z0-9]*\.[A-Za-z0-9\.]+)', str(str1))
    result = ''
    if len(it) > 0:
        result = it[0]
    if not result:
        result = 'unknown'
    return result


df['to_re'] = pd.Series(map(lambda str: email_re(str), df['to']))
df['from_re'] = pd.Series(map(lambda str: email_re(str), df['from']))
print(df.head())
