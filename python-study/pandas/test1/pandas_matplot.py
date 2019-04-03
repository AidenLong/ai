# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# # series 制图
# data1 = np.random.randint(700, 1000, 12)
# data2 = np.random.randint(800, 1000, 12)
#
# ser1 = pd.Series(data1, index=[str(i) + '月' for i in np.arange(1, 13)])  #
# ser2 = pd.Series(data2, index=[str(i) + '月' for i in np.arange(1, 13)])
# ser1.plot(label='鼓浪屿', style='r--o', yticks=(np.linspace(500, 1000, 5)), xticks=(np.arange(1, 13)))
# ser2.plot(label='张家界', style='g-o', yticks=(np.linspace(500, 1000, 5)))
#
# # 设置标签
# plt.xlabel('月份')
# plt.ylabel('人次/百万')
# plt.title('游客流量表')
#
# plt.legend()
# plt.show()

# data = np.random.randint(700, 1000, (12,3))
# df = pd.DataFrame(data,index=[str(i) + '月' for i in range(1, 13)], columns=['a','b','c'])
# df.plot(figsize=(8,5),style=['r-o','g--o','m-.o'], xticks=np.arange(1,13))
# plt.xlabel('月份')
# plt.ylabel('人次/百万')
# plt.title('游客流量表')
#
# plt.show()

# 柱状图
# data1 = np.random.randint(700, 1000, 12)
# data2 = np.random.randint(800, 1000, 12)
#
# ser1 = pd.Series(data1, index=[str(i) + '月' for i in np.arange(1, 13)])
# ser2 = pd.Series(data2, index=[str(i) + '月' for i in np.arange(1, 13)])
#
# ser1.plot(figsize = (10,8),kind = 'bar',color = 'lightskyblue',width = -0.2,align = 'edge')
# ser2.plot(figsize = (10,8),kind = 'bar',color = 'yellowgreen',width = 0.2,align = 'edge')
#
# plt.show()


data = np.random.randint(700,1000,(12,2))
df = pd.DataFrame(data,index =  [str(i)+'月' for i in np.arange(1,13)],columns = ['鼓浪屿','张家界'])
df.plot(kind = 'bar',figsize = (8,5),rot= 90*np.pi/180)

plt.show()