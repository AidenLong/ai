# -*- coding:utf-8 -*-

from sklearn import datasets, linear_model
import numpy as np
import matplotlib.pyplot as plt

# Lars算法的迭代步骤（系数从0开始迭代，直到残差足够小，或者所有维度都参与模型）
# 读取糖尿病数据集
diabetes = datasets.load_diabetes()
X = diabetes.data
Y = diabetes.target

# 函数说明
# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.lasso_path.html#sklearn.linear_model.lasso_path

# coordinate descent 方法系数路径
lasso_alphas, lasso_coefs, _ = linear_model.lasso_path(X, Y, verbose=True)
print("lasso_alphas.shape：", lasso_alphas.shape, "lasso_coefs.shape:", lasso_coefs.shape)

# lars方法系数路径
lars_alphas, _, lars_coefs = linear_model.lars_path(X, Y, method='lasso', verbose=True)
print("lars_alphas.shape：", lars_alphas.shape, "lars_coefs.shape:", lars_coefs.shape)

# 计算每一列的系数和（一次迭代得到的系数和），用于展现迭代次数对系数的影响
c_lars = np.sum(np.abs(lars_coefs), axis=0)
c_lasso = np.sum(np.abs(lasso_coefs), axis=0)
# 画图
plt.figure(figsize=(12, 10))
plt.subplot(2, 2, 1)
plt.plot(c_lars, lars_coefs.T, linestyle='--')
ymin, ymax = plt.ylim()
plt.xlabel('sum|coef|')
plt.ylabel('Coefficients')
plt.title('LARS Path')
plt.subplot(2, 2, 2)
plt.plot(c_lasso, lasso_coefs.T, linestyle='--')
ymin, ymax = plt.ylim()
plt.xlabel('sum|coef|')
plt.ylabel('Coefficients')
plt.title('LASSO-coordinate descent Path')

plt.axis('tight')
plt.show()


