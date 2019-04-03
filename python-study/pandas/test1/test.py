# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np

'''
    将附件中的数据导入DataFrame中，实现以下操作：
1，对异常数据（数值为0）进行值替换，替换为当前列的平均值；
2，对zwyx列的数据进行平均值统计，其他列做计数统计（提示使用value_counts()）；
3，得到zwmc字段的唯一值列表；
4，通过group函数，实现对于dd字段的分组，并按照城市计算每个城市的最大薪资，
    使用折线图，显示Top10城市。（选做）
'''
data = pd.read_csv('ca_list_copy.csv')
data.set_index('Id')
# print(data)

mean = data['zwyx'].mean()
new_data = data.replace({0.0, mean})
print(mean)
col = list(new_data.columns)
col.remove('Id')
col.remove('zwyx')
print(col)

new_dataFrame = pd.DataFrame([])
for i in col:
    cnt = new_data[i].value_counts()
    new_cnt = pd.DataFrame(cnt)
    new_cnt.columns = ['values']
    new_dataFrame = pd.concat([new_dataFrame,new_cnt])
print(new_dataFrame)
# new_data

# print(list(data['zwmc'].unique()))

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

city = data.groupby('dd')['zwyx'].max().sort_values(ascending=False).head(10)

city.plot()
plt.show()
