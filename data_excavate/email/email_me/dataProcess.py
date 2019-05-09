# -*-coding:utf-8 -*-
import os
import sys
import time


def make_label_dict(file_path):
    type_dict = {"spam": "1", "ham": "0"}
    index_file = open(file_path)  # ./date/full/index
    index_dict = {}
    try:
        for line in index_file:
            arr = line.split(" ")
            if len(arr) == 2:
                key, value = arr
                # (spam),(../data/000/000)
            # 添加到字段中
            value = value.replace("../data", "").replace("\n", "")
            # (spam),(/000/000)
            index_dict[value] = type_dict[key.lower()]
    finally:
        index_file.close()
    return index_dict


# 邮件的文件内容数据读取
def email_content_to_dict(file_path):
    './data/data/000/000'
    file = open(file_path, "r", encoding="gb2312", errors='ignore')
    content_dict = {}

    try:
        is_content = False  # 初始化为False后，在循环之外
        for line in file:
            line = line.strip()
            if line.startswith("From:"):
                # From: "yan"<(8月27-28,上海)培训课程>
                content_dict['from'] = line[5:]
            elif line.startswith("To:"):
                content_dict['to'] = line[3:]
            elif line.startswith("Date:"):
                content_dict['date'] = line[5:]
            elif not line:
                is_content = True

            # 处理邮件内容
            if is_content:
                if 'content' in content_dict:
                    content_dict['content'] += line
                else:
                    content_dict['content'] = line
    finally:
        file.close()
    return content_dict


# 邮件数据处理
def dict_to_str(file_path):
    content_dict = email_content_to_dict(file_path)

    # 进行处理
    result_str = content_dict.get('from', 'unkown').replace(',', '').strip() + ","
    result_str += content_dict.get('to', 'unknown').replace(',', '').strip() + ","
    result_str += content_dict.get('date', 'unknown').replace(',', '').strip() + ","
    result_str += content_dict.get('content', 'unknown').replace(',', ' ').strip()
    return result_str


# 使用函数开始数据处理
start = time.time()
index_dict = make_label_dict('../data/full/index')
# index_dict = 制作标签字典('C:\\Users/Administrator/Desktop/index')#('./data/full/index')
# print(index_dict)
# sys.exit(0)
list0 = os.listdir('../data/data')  # 文件夹的名称

for l1 in list0:  # 开始把N个文件夹中的file写入N*n个wiriter
    l1_path = '../data/data/' + l1
    # l1_path='../data/data/000'
    print('开始处理文件夹' + l1_path)
    list1 = os.listdir(l1_path)
    # list1文件列表

    write_file_path = '../data/process01_' + l1
    # 保存每个文件夹下面文件的文件 300行
    with open(write_file_path, "w", encoding='utf-8') as writer:
        for l2 in list1:
            l2_path = l1_path + "/" + l2  # 得到要处理文件的具体路径

            index_key = "/" + l1 + "/" + l2

            if index_key in index_dict:
                content_str = dict_to_str(l2_path)
                content_str += "," + index_dict[index_key] + "\n"
                writer.writelines(content_str)

with open('../data/result_process01', "w", encoding='utf-8') as writer:
    for l1 in list0:
        file_path = '../data/process01_' + l1
        print("开始合并文件：" + file_path)

        with open(file_path, encoding='utf-8') as file:
            for line in file:
                writer.writelines(line)

end = time.time()
print('数据处理总共耗时%.2f' % (end - start))
