# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np

# ser1 = pd.Series(np.random.randint(1,9,5),index = list('abcde'))
# print(ser1)
# # 序列排名
# print(ser1.rank())
# print(ser1.rank(method = 'max'))
# print(ser1.rank(method = 'min'))
# print(ser1.rank(method = 'first'))

df1 = pd.DataFrame(np.random.randint(1,9,(4,4)),index = list('abcd'),columns = list('abcd'))
print(df1)

# 按列排序
print(df1.rank())
# 按行排序
print(df1.rank(axis = 1))
