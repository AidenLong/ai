#-*- conding:utf-8 -*-

'''

'''


'''
    循环：
    while 循环 ：通过条件进行循环
    for：遍历一个序列

    跳转语句：
    break：直接跳出循环，不管条件是否为真，都不继续循环
    continue：跳出本次循环，直接开始下一次循环
'''


'''
    列表推导式
'''
# list01 = [1,2,3,4]  #声明列表
# list02 = list(range(6))
# print(list02)
# list01 = [1,2,3,4,3,5,6,7]
#
# list03 = []#声明一个空列表
# for i in range(3,10): #遍历数字序列
#     if i % 2 == 0:
#         list03.append(i) #将遍历得内容添加到列表当中
# print(list03)

# list03 = [i for i in range(3,10) if i % 2 == 0]  #简写方式
# print(list03)


'''
    字典推导
'''
list05 = ('joe','susan','black')
print(list(enumerate(list05)))

dict01 = {k:v for k,v in enumerate(list05)}
print(dict01)

# dict01 = {'name':'joe','age':'18'}
# print(list(dict01.items()))
# for k , v in dict01.items():
#     print(k,v)


'''
    嵌套列表推导
'''
names=[['Tom','Billy','Jefferson','Andrew','Wesley','Joe'],
       ['Alice','Jill','Ana','Wendy','Jennifer','Eva']]

# print(names[0][0])
# list04 = []
# for i in names:
#     for n in i :
#         list04.append(n)
# print(list04)

# lsit04 = [n for i in names for n in i if len(n) > 4]
# print(lsit04)




