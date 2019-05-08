# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('./household_power_consumption_1000.txt', sep=';', low_memory=False)
# 异常值替换
new_df = df.replace('?', np.nan)
# 有NaN时删除行
datas = new_df.dropna(0, how='any')

x = datas.iloc[:, 2:4]
y = datas.iloc[:, 5]

# 数据划分
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# 标准化
scaler = StandardScaler()
scaler.fit_transform(x_train)
scaler.transform(x_test)

# 模型训练
lr = LinearRegression()
lr.fit(x_train, y_train)

# 结果预测
y_predict = lr.predict(x_test)

# 结果评估
print(lr.score(x_test, y_test))

# 画图
x_len = np.arange(len(x_test))
plt.figure(figsize=(16, 8), facecolor='w')
plt.plot(x_len, y_test, 'r-', lw=2, label='真实值')
plt.plot(x_len, y_predict, 'g-', lw=2, label='预测值')
plt.legend(loc='upper right')
plt.title('功率与电流之间的关系，准确率 %.2f' % (lr.score(x_test, y_test) * 100))
plt.grid(True)
plt.show()
