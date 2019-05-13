# -*- coding:utf-8 -*-
# 导入包
import os
from surprise import Dataset, Reader
from surprise import BaselineOnly
from surprise import evaluate

# 加载数据 将数据转换为系数矩阵的形式
# 方式一，直接通过surprise的代码加载默认数据  下载电影数据，默认下载到当前用户根目录
# data = Dataset.load_builtin('ml-100k')
# 方式二：从文件中加载数据，要求文件中至少包含，用户id，商品id，评分
file_path = os.path.expanduser('./data/u.data')

# 制定一个数据读取器，比如给定数据格式，要求列名称不能改动（顺序可以变）
# line_format：给定数据中，每一行是按照什么顺序组合列信息的（和数据中的格式是一直的）
reader = Reader(line_format='user item rating timestamp', sep='\t')
data = Dataset.load_from_file(file_path, reader=reader)
print(data)

# 模型参数设置
# bsl_options = {
#     'method': 'als',  # 指定使用何种方式求解模型参数，默认是als，可选sgd
#     'n_epochs': 5,  # 指定迭代次数
#     'reg_i': 25,  # b_i计算过程中的正则化系数值，也就是ppt上的λ2
#     'reg_u': 10  # b_u计算过程中的正则化系数值，也就是ppt上的λ3
# }
bsl_options = {
    'method': 'sgd',  # 指定使用何种方式求解模型参数，默认是als，可选sgd
    'n_epochs': 50,  # 指定迭代次数
    'reg': 0.02,  # 正则化项系数，也就是ppt上的λ
    'learning_rate': 0.01  # 梯度下降中的学习率
}

# 模型构建，模型效果评估
# 1. 模型效果评估代码
# a. 做一个手动的数据分割操作（类似交叉验证）
# data.split(3)
#
# # b. 构建模型
#
# algo = BaselineOnly(bsl_options=bsl_options)
#
# # c. 模型效果评估运行
# evaluate(algo, data=data, measures=['rmse', 'mae', 'fcp'])

# 2. 模型构建代码
# 对训练数据进行构建（必须构建的，将读取的行数据转换为真实的稀疏矩阵形式）
# 在转换的过程中，会对用户id，商品id 进行重新序列编号
trainset = data.build_full_trainset()

# b. 构建模型
algo = BaselineOnly(bsl_options=bsl_options)

# c. 模型训练
algo.train(trainset)

# d. 模型预测
"""
在Surprise框架中，获取预测评分的API必须使用predict，predict底层会调用estimate API：
两个API的区别：
predict：传入的是实际的用户id和物品id，predict会处理负的评分(转换过程，也就是数据读取的反过程)
estimate: 传入的是转换之后的用户id和物品id，estimate不会处理
"""
uid = "253"
iid = "465"
pred = algo.predict(uid, iid, 5)
print("评分:{}".format(pred))
print("评分:{}".format(pred.est))
