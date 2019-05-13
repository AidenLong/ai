# -- encoding:utf-8 --

from surprise import Dataset
from surprise import SlopeOne
from surprise import evaluate

# 1. 加载数据
data = Dataset.load_builtin(name='ml-100k')

# 2. 将数据转换为Dataset训练集的形式
trainset = data.build_full_trainset()

# 3. 模型构建
algo = SlopeOne()

# 4. 模型效果评估
evaluate(algo=algo, data=data, measures=['rmse', 'mae', 'FCP'])

# 5. 模型训练
algo.fit(trainset)

# 6. 模型预测
print("用户1对于物品5的评分:{}".format(algo.predict("1", "5").est))
