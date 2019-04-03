import time
import pandas as pd
import numpy as np

# df=pd.DataFrame(np.random.randn(6,4),columns=list('ABCD'))
# df['F']=pd.date_range('20180115',periods=6)
#
# def getTime(x):
#     time_tup = time.strptime(str(x),'%Y-%m-%d %H:%M:%S')
#     return pd.Series(time_tup[0:3],index = ['年','月','日'])
#
#
# print(df['F'].apply(getTime))

'''
  请结合apply与groupy函数完成一下练习
  根据地区进行分组   查看平均年龄和工资
  根据年龄进行分组   查看平均工资
  根据性别进行分组   查看平均工资
  先根据地区，然后在根据性别进行分组，查看各地区不同性别的平均工资
'''
# df6 = pd.DataFrame({
#     'name':['joe', 'susan', 'anne', 'black', 'monika','ronaldo','leonarldo','tom','yilianna','bulanni'],
#     'age':[19,19,18,20,20,18,19,20,18,19],
#     'sex':['man','women','women','man','women','man','man','man','women','women'],
#     'address':['上海','北京','上海','北京','北京','上海','北京','上海','北京','上海'],
#     'money':[8000,8500,7000,9000,10000,7500,8800,9300,12000,11000]
# })
#
# print(df6)
# print(df6.groupby('address').apply(lambda x:x.mean()))
# print(df6.groupby('age')['money'].apply(lambda x:x.mean()))
# print(df6.groupby('sex')['money'].apply(lambda x:x.mean()))
# print(df6.groupby(['address','sex'])['money'].apply(lambda x:x.mean()))

df = pd.DataFrame({'性别' : ['男', '女', '男', '女',
                              '男', '女', '男', '男'],
                       '成绩' : ['优秀', '优秀', '及格', '差',
                              '及格', '及格', '优秀', '差'],
                       '年龄' : [15,14,15,12,13,14,15,16]})
GroupBy=df.groupby("性别")

# 根据性别进行成绩排序
# for name ,goru in GroupBy:
#     print(name)
#     print(goru.sort_values(by = '成绩'))

#根据性别进行成绩排序
# def s(x):
#     return x.sort_values(by = '成绩')
# df.groupby('性别').apply(s)
# print(df.groupby('性别').apply(lambda x:x.sort_values(by='成绩')))

total = df['性别'].count()
sex_count = df.groupby('性别')['性别'].count()
print(sex_count.apply(lambda x:str(x/total*100)+'%'))

score = df.groupby('成绩')['成绩'].count()
print(score.apply(lambda x : str(x)+'人'))
