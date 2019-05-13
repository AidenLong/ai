# -- encoding:utf-8 --

# 导入包
import os
import surprise
from surprise import Dataset
from surprise import KNNBasic, KNNWithMeans, KNNBaseline
from surprise import evaluate

"""
KNNBasic, KNNWithMeans, KNNBaseline:
KNNBasic: 最基本的协同过滤算法，就是直接对评分数据做计算操作
KNNWithMeans：在基本的协同过滤算法上，加入均值偏置转换
KNNBaseline：在基本的协同过滤算法上，将均值偏置转换为基准线偏置
"""

# 加载数据（将数据转换为按样本存储(user item rating)稀疏矩阵的形式）
# 1. 方式一：直接通过Surprise的代码加载默认数据
"""
会从网络上下载电影的评分数据，默认会下载到:~/.surprise_data文件夹中;
API: name参数可选值'ml-100k', 'ml-1m', and 'jester'.
"""
data = Dataset.load_builtin(name='ml-100k')

# 模型构建、模块的效果评估
# 1. 模型效果评估代码
# a. 做一个手动的数据分割操作（类似交叉验证）
data.split(5)

# b. 构建模型
sim_options = {'name': 'cosine', 'user_based': False}
algo = KNNBaseline(sim_options=sim_options, k=40)

# c. 模型效果评估的运行
evaluate(algo, data=data, measures=['RMSE', 'MAE', 'FCP'])
