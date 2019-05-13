# -*- coding:utf-8 -*-
from surprise import Dataset
from surprise import KNNBaseline
from six import iteritems
import surprise

data = Dataset.load_builtin(name='ml-100k')

trainset = data.build_full_trainset()

sim_options = {
    'name': 'cosine',
    'user_based': False
}

bsl_options = {
    'method': 'sgd',  # 指定使用何种方式求解模型参数，默认是als，可选sgd
    'n_epochs': 50,  # 指定迭代次数
    'reg': 0.02,  # 正则化项系数，也就是ppt上的λ
    'learning_rate': 0.01  # 梯度下降中的学习率
}
algo = KNNBaseline(sim_options=sim_options, bsl_options=bsl_options)

algo.fit(trainset)

dump_model = False
if dump_model:
    surprise.dump.dump('./result/itemcf.model', algo=algo)
else:
    with open('./result/user_item_rating.txt', 'w') as write_file:
        # 直接持久化预测结果
        all_inner_user_item_rating = trainset.ur
        all_inner_item_user_rating = trainset.ir
        user = 0
        for inner_user, _ in iteritems(all_inner_user_item_rating):
            row_uid = trainset.to_raw_uid(inner_user)
            for inner_item, _ in iteritems(all_inner_item_user_rating):
                row_itemid = trainset.to_raw_iid(inner_item)
                predict_rating = algo.predict(row_uid, row_itemid).est
                # print("用户%s对于物品%s的评分为：%s" % (row_uid, row_itemid, predict_rating))
                write_file.writelines('%s\t%s\t%0.3f\n' % (row_uid, row_itemid, predict_rating))
            user += 1
            print(user)
