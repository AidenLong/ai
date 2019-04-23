# -*- coding:utf-8 -*-
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import sklearn
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model.coordinate_descent import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False
# 拦截异常
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)


def notEmpty(s):
    return s != ''


# 加载数据
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
path = "datas/boston_housing.data"
# 由于数据文件格式不统一，所以读取的时候，先按照一行一个字段属性读取数据，然后再安装每行数据进行处理
fd = pd.read_csv(path, header=None)
print(fd.head())
# print (fd.shape)
data = np.empty((len(fd), 14))  # enumerate生成一列索 引i,d为其元素
for i, d in enumerate(fd.values):
    d = map(float, filter(notEmpty, d[0].split(' ')))  # filter一个函数，一个list
    # 根据函数结果是否为真，来过滤list中的项。
    data[i] = list(d)

# 分割数据
x, y = np.split(data, (13,), axis=1)
print(x[0:5])
y = y.ravel()  # 转换格式 拉直操作
ly = len(y)
print('样本数据量：%d，特征个数：%d' % x.shape)
print('target样本数据量：%d' % y.shape)

# Pieline 常用与并行调参
models = [
    Pipeline([
        ('ss', StandardScaler()),
        ('poly', PolynomialFeatures()),
        ('linear', RidgeCV(alphas=np.logspace(-3, 1, 20)))
    ]),
    Pipeline([
        ('ss', StandardScaler()),
        ('poly', PolynomialFeatures()),
        ('linear', LassoCV(alphas=np.logspace(-3, 1, 20)))
    ])
]

# 参数字典，字典中的key是属性的名字，value是可选的参数列表
parameters = {
    'poly__degree': [3, 2, 1],
    'poly__interaction_only': [True, False],  # 不产生交互项，如X1*X1
    'poly__include_bias': [True, False],  # 多项式幂为零的特征作为线性模型中的截距
    'linear__fit_intercept': [True, False]
}

rf = PolynomialFeatures(2, interaction_only=True)
a = pd.DataFrame({
    'name': [1, 2, 3, 4, 5],
    'score': [2, 3, 4, 4, 5]
})
b = rf.fit_transform(a)
print(b)

# 数据分割
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Lasso和Ridge模型比较运行图标展示
titles = ['Ridge', 'Lasso']
colors = ['g-', 'b-']
plt.figure(figsize=(16, 8), facecolor='w')
ln_x_test = range(len(x_test))

plt.plot(ln_x_test, y_test, 'r-', lw=2, label='真实值')
for t in range(2):
    # 获取模型并设置参数
    # GridSearchCV: 进行交叉验证，选择出最优的参数值出来
    # 第一个输入参数：进行参数选择的模型，param_grid： 用于进行模型选择的参数字段，要求是字典类型；cv: 进行几折交叉验证
    model = GridSearchCV(models[t], param_grid=parameters, cv=5, n_jobs=1)
    # 模型训练-网格搜索
    model.fit(x_train, y_train)
    # 模型效果值获取（最优参数）
    print("%s算法:最优参数:" % titles[t], model.best_params_)
    print("%s算法:R值=%.3f" % (titles[t], model.best_score_))
    # 模型预测
    y_predict = model.predict(x_test)
    # 画图
    plt.plot(ln_x_test, y_predict, colors[t], lw=t + 3, label=u'%s算法估计值,$R^2$=%.3f' % (titles[t], model.best_score_))
# 图形显示
plt.legend('upper left')
plt.grid(True)
plt.title('波士顿房屋价格预测')
plt.show()

# 模型训练===》单个Lasso模型（一阶特征选择）<2参数给定1阶情况的最优参数>
model = Pipeline([
    ('ss', StandardScaler()),
    ('poly', PolynomialFeatures(degree=1, include_bias=True, interaction_only=True)),
    ('linear', LassoCV(alphas=np.logspace(-3, 1, 20), fit_intercept=False))
])

# 模型训练
model.fit(x_train, y_train)

# 参数输出
print("参数:", list(zip(names, model.get_params('linear')['linear'].coef_)))
print("截距:", model.get_params('linear')['linear'].intercept_)

# L1-norm是可以做特征选择的，主要原因在于：通过Lasso模型训练后，
# 有的参数是有可能出现为0的或者接近0的情况； 针对于这一批的特征，可以进行特征的删除操作
# df.drop(xx)
# NOTE: 自己实现参数值绝对值小于10^-1的所有特征属性删除；要求：不允许明确给定到底删除那一个特征属性
# df.drop(['CHAS', 'DIS']) ==> 不允许这样写
# 实际工作中，一般情况下，除非低于10^-6才会删除
