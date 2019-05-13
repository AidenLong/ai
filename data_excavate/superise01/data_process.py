# -*- coding:utf-8 -*-

'''
电影数据的处理（将用户数据、电影数据以及用户电影评分数据合并）
'''

import pandas as pd

rating_col_names = ['user_id', 'item_id', 'rating', 'timestamp']
rating_df = pd.read_csv('./data/u.data', sep='\t', header=None, names=rating_col_names)
rating_df = rating_df.drop('timestamp', axis=1)
# print(rating_df.head())
# print(rating_df.info())
# print(rating_df.rating.value_counts())

user_col_names = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
user_df = pd.read_csv('./data/u.user', sep='|', header=None, names=user_col_names)
user_df = user_df.drop('zip_code', axis=1)
# print(user_df.head())
# print(user_df.info())
# print(user_df.gender.value_counts())
# print(user_df.occupation.value_counts())

item_col_names = ['item_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action',
                  'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                  'Film_Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci_Fi', 'Thriller', 'War', 'Western']
item_df = pd.read_csv('./data/u.item', sep='|', header=None, names=item_col_names)
item_df = item_df.drop(['movie_title', 'release_date', 'video_release_date', 'IMDb_URL'], axis=1)
# print(item_df.head())
# print(item_df.info())

result = pd.merge(user_df, rating_df, on='user_id')
result = pd.merge(result, item_df, on='item_id')
# print(result.head())
# print(result.info())

# 字符串数据的处理，亚编码
result = pd.get_dummies(result)
print(result.info())

# 处理好的数据保存
result.to_csv('./data/merge_user_item_rating.csv', sep=',', index=False)
