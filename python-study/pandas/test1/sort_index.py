# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd

# ser1 = pd.Series(np.random.randint(1, 9, 4), index=['a', 'b', 'c', 'd'])
# print(ser1)
#
# print(ser1.sort_index())
# print(ser1.sort_index(ascending=False))

df1 = pd.DataFrame(np.random.randint(1, 9, (4, 4)), index=list('BCDA'), columns=list('bcda'))
print(df1)

print(df1.sort_index())
print(df1.sort_index(ascending=False))
print(df1.sort_index(axis=1))
print(df1.sort_index(axis=1, ascending=False))
