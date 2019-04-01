#-*- conding:utf-8 -*-
''''''
'''
    函数，计算传入字符串的个数
'''
# def getLen(s):
#     if isinstance(s,str): #args1: 数据 args2：数据类型
#         return  len(s)
#     else:
#         return '类型错误'
# num = getLen('sdfdsf sdf')
# print(num)


'''
    函数，判断用户传入的字符串、列表、元组长度是否大于5
'''
# def getLen2(args):
#     if isinstance(args,(str,list,tuple)):
#         if len(args) > 5:
#             print('传入的对象长度大于5')
#         else:
#             print('传入的对象长度小于5')
#     else:
#         print('类型有误')
#
# getLen2(['a','b',1,2])


'''
    写入不定个数的字符串拼接第一个和最后一个字符串
'''
# def getStr(*args):
#     return args[0]+args[-1]
#
# print(getStr('1','2','3'))

'''
    传入多个参数，以list返回
'''
# def getList(*args):
#     li = []
#     for i in args:
#         li.append(i)
#     return li
#
# list01 = getList(1,2,3,4,5)
# print(list01)

'''
    定义一个函数，输入不定个数的数字，返回所有数字的和
'''
# def getSum(*args):
#     x = 0
#     for i in args:
#         x+=i
#     return x
#
# num = getSum(1,2,3,4,5)
# print(num)



# def f(x,y):
# 	l = [i for i in range(1,x+1)]
# 	t = 0
# 	while(len(l)!=2):
# 		n = len(l)
# 		l.remove(l[(y-1+t)%len(l)])
# 		t = (y-1+t)%n
# 	print(l)
#
# f(41,3)


'''
    希腊打仗，39犹太人，约瑟夫 和他的朋友一起，41人。
    41一个围城一个圈。从一开始报数，谁报到三，谁就退出游戏
    16 31

    n = 9 总人数
    m = 3 报数
    k  = (k+(m-1))%len(list)   索引
'''

# def func(n,m):
#     #生成一个列表
#     people = list(range(1,n+1))
#     k = 0 #定义开始索引
#
#     #开始循环报数
#     while len(people) > 2:
#         k = (k+(m-1))%len(people)
#         print('kill:',people[k])
#         del(people[k])
#         print(k)
#     return people
#
# print(func(41,3))

'''
定义一个函数，实现两个数四则运算，要注意有3个参数，分别是运算符和两个用于运算的数字。
'''
def calculation(a1, a2, b):
    if b == '+':
        return a1 + a2
    elif b == '-':
        return a1 - a2
    elif b == '*':
        return a1 * a2
    elif b == '/':
        return a1 / a2
    else:
        return '元算符错误'

print(calculation(1, 2, '+'))
print(calculation(1, 2, '*'))
print(calculation(1, 2, '-'))
print(calculation(1, 2, '!'))