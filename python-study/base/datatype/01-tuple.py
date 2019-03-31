#_*_ conding:utf-8 _*_
'''

    创建元组   不可变
'''
tup01 = (1,2,3,4,5)
# print(tup01[1])
#
# tup01[0] = 9
# print(tup01)

#删除 元组
del tup01[0]
del tup01   #彻底删除
print(tup01)

#获取元组的长度
print(len(tup01))

l = ['a','b','c']
print(type(l))
l=tuple(l)   #没有返回值
print(type(l))   #转完之后是一个元组

l = list(l)     #将元组转换回列表
print(type(l))




