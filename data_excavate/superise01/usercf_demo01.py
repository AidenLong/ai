# -*- coding:utf-8 -*-
# 导入包
import os
from surprise import Dataset
from surprise import KNNBaseline, KNNWithMeans, KNNBasic
from surprise import evaluate

"""
KNNBasic, KNNWithMeans, KNNBaseline:
KNNBasic: 最基本的协同过滤算法，就是直接对评分数据做计算操作
KNNWithMeans：在基本的协同过滤算法上，加入均值偏置转换
KNNBaseline：在基本的协同过滤算法上，将均值偏置转换为基准线偏置
"""
# 加载数据 将数据转换为系数矩阵的形式
# 方式一，直接通过surprise的代码加载默认数据  下载电影数据，默认下载到当前用户根目录
data = Dataset.load_builtin('ml-100k')

# 1. 模型效果评估代码
# a. 做一个手动的数据分割操作（类似交叉验证）
# data.split(3)
#
# # b. 构建模型
# sim_options = {'name': 'pearson', 'user_based': True}
# # algo = KNNBasic(sim_options=sim_options)
# # algo = KNNWithMeans(sim_options=sim_options)
# algo = KNNBaseline(sim_options=sim_options)
#
# # c. 模型效果评估运行
# evaluate(algo, data=data, measures=['rmse', 'mae', 'fcp'])

# 2. 模型构建代码
# 对训练数据进行构建（必须构建的，将读取的行数据转换为真实的稀疏矩阵形式）
# 在转换的过程中，会对用户id，商品id 进行重新序列编号
trainset = data.build_full_trainset()

# b. 构建模型
sim_options = {'name': 'jaccard', 'user_based': True}
# algo = KNNBasic(sim_options=sim_options)
# algo = KNNWithMeans(sim_options=sim_options)
algo = KNNBaseline(sim_options=sim_options)

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
