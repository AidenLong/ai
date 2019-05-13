# -- encoding:utf-8 --

from six import iteritems
import surprise
from surprise import Dataset
from surprise import KNNBaseline

# 1. 读取数据
data = Dataset.load_builtin(name='ml-100k')

# 2. 模型训练
# 2.1 数据稀疏化矩阵构建
trainset = data.build_full_trainset()

# 2.2 构建模型
sim_options = {
    'name': 'cosine',  # 使用余弦相似度计算
    'user_based': False  # 设置是采用UserCF，还是ItemCF；设置为True表示使用UserCF，设置为False表示使用ItemCF
}
bsl_options = {
    'method': 'sgd',  # 指定使用何种方式求解模型参数，默认是als，可选sgd
    'n_epochs': 50,  # 指定迭代次数
    'reg': 0.02,  # 正则化项系数，也就是ppt上的λ
    'learning_rate': 0.01  # 梯度下降中的学习率
}
algo = KNNBaseline(sim_options=sim_options, bsl_options=bsl_options)

# 2.3 模型训练
algo.fit(trainset)

# 2.4 保存物品的相似度矩阵
file_path = './result/item_sim.data'
with open(file_path, mode='w') as writer:
    n_items = algo.sim.shape[0]
    for i in range(n_items):
        row_item_i = algo.trainset.to_raw_iid(i)
        for j in range(n_items):
            if i != j:
                row_item_j = algo.trainset.to_raw_iid(j)
                sim = algo.sim[i][j]
                writer.writelines('%s\t%s\t%.3f\n' % (row_item_i, row_item_j, sim))
