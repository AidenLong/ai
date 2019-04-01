# -*- coding:utf-8 -*-
'''
    lambda [参数]:  表达式  默认返回
'''
#没有参数的匿名函数
# s = lambda : 'hahaha'
# print(s())  #调用

#有参数的匿名函数
# s = lambda x,y: x+y
# print(s)

#矢量化三元运算符
# s = lambda x,y: 'hahah' if x>2 else y
# print(s(3,2))

#字典排序
dic = {'a':1,'c':2,'b':3}
# dic.sort()#没有sort方法

#通过内置函数 sorted进行排序
# dic = sorted(dic.items(),reverse= True)   #内置函数有返回值  默认通过KEY排序
# print({k:v for k,v in dic})  #字典推导式
# print(dic.items())
# dic = sorted(dic.items(),key = lambda x:x[1],reverse= True)   #内置函数有返回值  默认通过KEY排序
# print(dic)
# print({k:v for k,v in dic})#字典推导式


# lsit01 = [
#     {'name':'joe','age':18},
#     {'name':'susan','age':19},
#     {'name':'tom','age':17}
# ]

# print(dic.items())
# dic = sorted(lsit01,key = lambda x:x['name'],reverse=True)   #内置函数有返回值  默认通过KEY排序
# print(dic)


'''
    高阶函数
'''
# list01 = [1,3,5,7,9]
#
#map
# list02 = [2,4,6,8,10]
# new_list = map(lambda x,y:x*y,list01,list02)
# print(list(new_list)) #将map对象转换为list
# for i in new_list:
#     print(i)

#filter
# list02 = [2,4,6,8,10]
# new_list = filter(lambda x:x>4,list02)
# print(list(new_list))

# for i in new_list:
#     print(i)

#reduce
from functools import reduce
# list02 = [2,4,6,8,10]
# new_list = reduce(lambda x,y:x+y,list02,0)
# print(new_list)


'''
    示例
'''
name = ['joe','susan','black','lili']
age = [18,19,20,21]
sex = ['m','w','m','w']

#格式化用户的英文名，要求首字母大写，其它字母小写
# new_name = map(lambda x:x.title() ,name)
# new_name = list(new_name)
# print(new_name)

#将用户英文名、年龄、性别三个集合的数据结合到一起，形成一个元祖列表
# users = map(lambda x,y,z:(x,y,z),name,age,sex)
# new_users = list(users)
# # print(new_users)
# #过滤性别为男的用户
# new_users = filter(lambda x:x[2] == 'm',new_users)
# new_users = list(new_users)
# print(new_users)

#求性别为男的用户的平均年龄
# total_age = reduce(lambda x,y:x[1]+y[1],new_users)
# print(total_age/len(new_users))
