# -*- coding:utf-8 -*-
import warnings

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.exceptions import DataConversionWarning

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import time


# 创建一个函数格式化时间字符串
def date_format(dt):
    t = time.strptime(' '.join(dt), '%d/%m/%Y %H:%M:%S')
    return (t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)


# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False
## 拦截异常
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

# 加载数据
path = 'datas/household_power_consumption_1000.txt'
df = pd.read_csv(path, sep=";", low_memory=False)

# 事件、有功功率、无功功率、电压、电流、厨房用电功率、洗衣机用电功率、热水器用电功率
names2 = df.columns
names = ['Date', 'Time', 'Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity',
         'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

# 异常数据处理
new_df = df.replace('?', np.nan)
datas = new_df.dropna(axis=0, how='any')  # 只要有数据为空，就进行行删除操作

# 时间和电压之间的关系（Linear）
# 获取x和y变量，并将时间转换为数值型连续变量
X = datas[names[0:2]]
X = X.apply(lambda x: pd.Series(date_format(x)), axis=1)
Y = datas[names[4]].values

# 对数据集进行测试集训练集划分
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# 数据标准化
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# 模型训练
lr = LinearRegression()
lr.fit(X_train, Y_train)

# 模型校验
y_predict = lr.predict(X_test)

# 模型效果
print('准确率:', lr.score(X_test, Y_test))

# 预测值和实际值画图比较
t = np.arange(len(X_test))
plt.figure(facecolor='w')
plt.plot(t, Y_test, 'r-', linewidth=2, label=u'真实值')
plt.plot(t, y_predict, 'g-', linewidth=2, label=u'预测值')
plt.legend(loc='lower right')
plt.title(u"线性回归预测时间和功率之间的关系", fontsize=20)
plt.grid(b=True)  # 网格
plt.show()

# 时间和电压之间的关系(Linear-多项式)
models = [
    Pipeline([
        ('Poly', PolynomialFeatures()),  # 给定进行多项式扩展操作
        ('Linear', LinearRegression(fit_intercept=False))
    ])
]
model = models[0]
# 获取x和y变量,并将时间转换为数值型连续变量
X = datas[names[0:2]]
X = X.apply(lambda x: pd.Series(date_format(x)), axis=1)
Y = datas[names[4]]

# 对数据集进行测试集训练集划分
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# 数据标准化
ss = StandardScaler()
X_train = ss.fit_transform(X_train)  # 训练并转换
X_test = ss.transform(X_test)  # 直接使用在模型构建数据上进行一个数据标准化操作

# 模型训练
t = np.arange(len(X_test))
N = 5
d_pool = np.arange(1, N, 1)  # 阶
m = d_pool.size
clrs = []  # 颜色
for c in np.linspace(16711680, 255, m):
    clrs.append('#%06x' % int(c))
line_width = 3

plt.figure(figsize=(12, 6), facecolor='w')
for i, d in enumerate(d_pool):
    plt.subplot(N - 1, 1, i + 1)
    plt.plot(t, Y_test, 'r-', label=u'真实值', ms=10, zorder=N)
    model.set_params(Poly__degree=d)  # 设置多项式的阶乘
    model.fit(X_train, Y_train)
    lin = model.get_params('Linear')['Linear']
    output = u'%d阶,系数为：' % d
    if hasattr(lin, 'alpha_'):
        idx = output.find(u'系数')
        output = output[:idx] + (u'alpha=%.0f，' % lin.alpha_) + output[idx:]
    if hasattr(lin, 'l1_ratio_'):
        idx = output.find(u'系数')
        output = output[:idx] + (u'l1_ratio=%.6f, ' % lin.l1_ratio_) + output[idx:]
    print(output, lin.coef_.ravel())

    y_hat = model.predict(X_test)
    s = model.score(X_test, Y_test)

    z = N - 1 if (d == 2) else 0
    label = u'%d阶，准确率=%.3f' % (d, s)
    plt.plot(t, y_hat, color=clrs[i], lw=line_width, alpha=0.75, label=label, zorder=z)
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.ylabel(u'%d阶结果' % d, fontsize=12)

# 预测值和实际值画图比较
plt.suptitle(u"线性回归预测时间和功率之间的多项式关系", fontsize=20)
plt.grid(b=True)
plt.show()
