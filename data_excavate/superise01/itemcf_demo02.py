# -- encoding:utf-8 --

import surprise

# 1. 加载保存好的模型
algo = surprise.dump.load('./result/itemcf.model')[1]

# 2. 使用加载的模型对数据做一个预测
row_user_ids = ['196', '2', '3']
row_item_ids = ['242', '6', '7', '8']
for user_id in row_user_ids:
    for item_id in row_item_ids:
        print("用户%s对于物品%s的评分为:%.3f" % (user_id, item_id, algo.predict(user_id, item_id).est))
