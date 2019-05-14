# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False


'''
替换
df.int_rate = pd.Series(df.int_rate).str.replace('%', '').astype(float)
选择object类型的列
df.select_dtypes(include=['object']).columns
替换
df['emp_length'].fillna(value=0, inplace=True)
添加一列
df.select_dtypes(include=['O']).describe().T.assign()
相关性、相关程度
cor = df.corr()
cor.loc[:, :] = np.tril(cor, k=-1)
cor = cor.stack()
print(cor[(cor > 0.7) | (cor < -0.7)])
亚编码
df = pd.get_dummies(df)
'''
df = pd.read_csv('./data/LoanStats3a.csv', skiprows=1, low_memory=False)
print(df.head(5))

print(df['loan_status'].value_counts())
# 删除id，member_id
df.drop('id', 1, inplace=True)
df.drop('member_id', 1, inplace=True)
# 修改int_rate

df.int_rate = pd.Series(df.int_rate).str.replace('%', '').astype(float)
# 删除全为NaN的行和列
df.dropna(axis=0, how='all', inplace=True)
df.dropna(axis=1, how='all', inplace=True)

print(df.emp_title.value_counts().head())
# emp_title种类太多，删除
print(df.emp_title.unique().shape)  # 30658
df.drop(['emp_title'], 1, inplace=True)

print(df.emp_length.value_counts())
df.replace('n/a', np.nan, inplace=True)
df['emp_length'].fillna(value=0, inplace=True)
df['emp_length'].replace(to_replace='[^0-9]+', value='', inplace=True, regex=True)
df['emp_length'] = df['emp_length'].astype(int)
print(df.emp_length.value_counts())

# 计算object类型value值数量
for col in df.select_dtypes(include=['object']).columns:
    print('Columns {} has {} unique instances'.format(col, len(df[col].unique())))
    if len(df[col].unique()) > 30:
        df.drop([col], 1, inplace=True)

# 查看缺省值所占比例问题
print(df.select_dtypes(include=['O']).describe().T.assign(
    missing_pct=df.apply(lambda x: (len(x) - x.count()) / float(len(x)))))

df.drop(['debt_settlement_flag_date', 'settlement_status'], 1, inplace=True)

# 贷后相关的字段删除
df.drop(['out_prncp', 'out_prncp_inv', 'total_pymnt',
         'total_pymnt_inv', 'total_rec_prncp', 'grade'], 1, inplace=True)
df.drop(['total_rec_int', 'total_rec_late_fee',
         'recoveries', 'collection_recovery_fee',
         'collection_recovery_fee'], 1, inplace=True)
df.drop(['last_pymnt_amnt'], 1, inplace=True)
df.drop(['policy_code'], 1, inplace=True)

# 处理缺省值的情况
print(df.select_dtypes(include=['float']).describe().T.assign(
    missing_pct=df.apply(lambda x: (len(x) - x.count()) / float(len(x)))))
# 删除缺失率比较高的特征
df.drop(['settlement_amount', 'settlement_percentage', 'settlement_term', 'mths_since_last_record'], 1, inplace=True)

# 处理缺省值的情况
print(df.select_dtypes(include=['int']).describe().T.assign(
    missing_pct=df.apply(lambda x: (len(x) - x.count()) / float(len(x)))))

# 查看最终目标属性（需要预测的指标）
print(df['loan_status'].value_counts())

# 替换目标属性的值，1表示好用户，0表示坏用户
df.loan_status.replace('Fully Paid', int(1), inplace=True)
df.loan_status.replace('Charged Off', int(0), inplace=True)
df.loan_status.replace('Does not meet the credit policy. Status:Fully Paid', np.nan, inplace=True)
df.loan_status.replace('Does not meet the credit policy. Status:Charged Off', np.nan, inplace=True)
print(df['loan_status'].value_counts())

# 计算关联信息
cor = df.corr()
cor.loc[:, :] = np.tril(cor, k=-1)
cor = cor.stack()
print(cor[(cor > 0.7) | (cor < -0.7)])

# 删除相关性比较强的特征属性： 因为如果存在多个特征之间是强相关的，那么其实可以用其中任意一个特征即即可得到特征属性和y值之间的映射关系
df.drop(['funded_amnt', 'funded_amnt_inv', 'installment'], 1, inplace=True)

# 查看一下各个特征属性取值为空的样本数目
print(df.isnull().sum())
# 1. 数值为空的进行填充，填充要不填充默认值，0/1; 要不中值，均值
df.fillna(0.0, inplace=True)
df.fillna(0, inplace=True)

# 哑扁码操作
df = pd.get_dummies(df)
print(df.info())

print(df.head(1).values)

# 模型的输出
# df.to_csv('../data/features01.csv', header=True, index=False)
