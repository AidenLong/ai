# -- encoding:utf-8 --

import surprise
from surprise import Dataset
from surprise import SVD
import numpy

# 1. 加载数据
data = Dataset.load_builtin(name='ml-100k')

# 2. 数据转换
trainset = data.build_full_trainset()

# 3. 模型构建
algo = SVD(n_factors=50, n_epochs=10, reg_all=0.2, lr_all=0.05)

# 4. 模型训练
algo.fit(trainset)

# 5. 模型应用
surprise.dump.dump('./result/svd.model')
print("用户1对于物品5的评分:{}".format(algo.predict("1", "5")))
