# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np

data = pd.DataFrame({
    'key1':list('aabba'),
    'key2': ['one','two','one','two','one'],
    'key3':list('abcde'),
    'data1': np.random.randint(1,9,5),
    'data2': np.random.randint(1,9,5),
    'data3': np.random.randint(1,9,5)
})

# print(data)

# datas = data.groupby('key1')
# print(list(datas))
#
# for name,gro in datas:
#     print(name)
#     print(gro)

#随机下标
# key = [1,2,1,1,2]
# print(list(data.groupby(key)))
# print(data.groupby(key)['data1'].mean())

#聚合
# print(data.groupby('key1').sum())
# print(data.groupby('key1').mean())
# print(data.groupby('key1').max())
# print(data.groupby('key1').min())

# print(list(data.groupby(['key1','key2'])))

data = np.random.randint(1,10,(4,4))
df1 = pd.DataFrame(data,columns = ['a','b','c','d'])
print(df1)

#自定义聚合
print(df1.apply(lambda x:x*10))
print(df1['a'].apply(lambda x:x*10))
print(df1.apply(lambda x:x*10,axis = 1))
print(df1.loc[1:2].apply(lambda x:x*10,axis = 1))

