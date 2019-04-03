# -*- coding:utf-8 -*-
# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd

# ser1 = pd.Series(np.random.randint(1, 9, 4), index=['a', 'b', 'c', 'd'])
# print(ser1)
#
# print(ser1.sort_values())
# print(ser1.sort_values(ascending=False))

df1 = pd.DataFrame(np.random.randint(1, 9, (4, 4)), index=list('BCDA'), columns=list('bcda'))
print(df1)

# 默认升序 按列
print(df1.sort_values(by='a'))
# 降序排序
print(df1.sort_values(by='a',ascending=0))
# 默认升序 按行
print(df1.sort_values(by='A',axis=1))
print(df1.sort_values(by='A',axis=1, ascending=0))
