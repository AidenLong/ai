# -*- coding:utf-8 -*-
# 引入所需要的全部包
# 01_家庭用电预测：线性回归算法(时间与功率&功率与电流之间的关系)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import DataConversionWarning
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
import time
import warnings

## 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False
## 拦截异常
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

# 加载数据
# 日期、
# 事件、有功功率、无功功率、电压、电流、厨房用电功率、洗衣机用电功率、热水器用电功率
path = 'datas/household_power_consumption_1000.txt'
# 没有混合类型时通过low_memory=F调用更多内存，加快效率
df = pd.read_csv(path, sep=";", low_memory=False)

# 获取前5行数据
print(df.head())
# 查看格式信息
print(df.info())

# 异常数据处理（异常数据过滤）
new_df = df.replace('?', np.nan)  # 替换非法字符为np.nan
datas = new_df.dropna(axis=0, how='any')  # 只要有一个数据为空，就进行行删除操作
print(datas.describe().T)  # 观察数据的多种统计指标


# 创建一个时间函数格式化字符串
def date_format(dt):
    t = time.strptime(' '.join(dt), '%d/%m/%Y %H:%M:%S')
    return (t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)


# 获取x和y变量，并将时间转换为数值类型连续变量
X = datas.iloc[:, 0:2]
X = X.apply(lambda x: pd.Series(date_format(x)), axis=1)
Y = datas['Global_active_power']

print(X.head(2))

'''
    对数据集进行测试集合训练集划分
    X：特征矩阵（类型一般是DataFrame）
    Y：特征矩阵对应的Label（类型一般是Series）
    test_size：对X/Y进行划分的时候，测试集合的数据占比，是一个（0-1）之间的float类型的值
    random_state：数据分割是基于随机器进行分割的,该参数给定随机数种子；
        给一个值(int类型)的作用就是保证每次分割所产生的数数据集是完全相同的
'''
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(X_train.describe().T)

'''
    数据标准化
    StandardScaler：将数据转换为标准差为1的数据集（有一个数据的映射）
    如果一个API名字有fit，那么就有模型训练的含义
    如果一个API名字中有transform， 那么就表示对数据具有转换的含义操作
    如果一个API名字中有predict，那么就表示进行数据预测，会有一个预测结果输出
'''
ss = StandardScaler()
X_train = ss.fit_transform(X_train)  # 训练并转换
X_test = ss.transform(X_test)  # 直接使用在模型构建数据上进行一个数据标准化操作

print(pd.DataFrame(X_test).describe().T)

# 模型训练
lr = LinearRegression()
lr.fit(X_train, Y_train)  # 训练模型
# 模型校验
y_predict = lr.predict(X_test)  # 预测结果

print('训练R2：', lr.score(X_train, Y_train))
print('测试R2：', lr.score(X_test, Y_test))
mse = np.average((y_predict - Y_test) ** 2)
rmse = np.sqrt(mse)
print('rmse', rmse)

## 模型保存/持久化
# 在机器学习部署的时候，实际上其中一种方式就是将模型进行输出；另外一种方式就是直接将预测结果输出
# 模型输出一般是将模型输出到磁盘文件
from sklearn.externals import joblib

joblib.dump(ss, 'data_ss.model')  # 将标准化模型保存
joblib.dump(lr, 'data_lr.model')  # 将模型保存

ss = joblib.load('data_ss.model')
lr = joblib.load('data_lr.model')

# 使用加载的模型进行预测
data1 = [[2006, 12, 17, 12, 25, 0]]
data1 = ss.transform(data1)
print(data1)
print(lr.predict(data1))

## 预测值和实际值画图比较
t = np.arange(len(X_test))
plt.figure(facecolor='w')  # 建一个画布，facecolor是背景色
plt.plot(t, Y_test, 'r-', linewidth=2, label='真实值')
plt.plot(t, y_predict, 'g-', linewidth=2, label='预测值')
plt.legend(loc='upper left')  # 显示图例，设置图例的位置
plt.title("线性回归预测时间和功率之间的关系", fontsize=20)
plt.grid(b=True)  # 加网格
plt.show()

'''
    功率和电流之间的关系
'''
X = datas.iloc[:, 2:4]
Y2 = datas.iloc[:, 5]

X2_train, X2_test, Y2_train, Y2_test = train_test_split(X, Y2, test_size=0.2, random_state=1)

# 数据归一化
scaler2 = StandardScaler()
X2_train = scaler2.fit_transform(X2_train)  # 训练并转换
X2_test = scaler2.transform(X2_test)  ## 直接使用在模型构建数据上进行一个数据标准化操作

# 模型训练
lr2 = LinearRegression()
lr2.fit(X2_train, Y2_train)

# 结果预测
Y2_predict = lr2.predict(X2_test)

# 模型评估
print("电流预测准确率: ", lr2.score(X2_test, Y2_test))
print("电流参数:", lr2.coef_)

# 绘制图表
# 电流关系
t = np.arange(len(X2_test))
plt.figure(facecolor='w')
plt.plot(t, Y2_test, 'r-', linewidth=2, label=u'真实值')
plt.plot(t, Y2_predict, 'g-', linewidth=2, label=u'预测值')
plt.legend(loc='lower right')
plt.title(u"线性回归预测功率与电流之间的关系", fontsize=20)
plt.grid(b=True)
plt.show()
