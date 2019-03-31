# -*- coding:utf-8 -*-

'''
数据类型
    Number:   0 为false
    字符串:
    列表
    元祖
    字典
'''

'''
    int
'''

# a = 1
# print(type(a)) #type是python中一个用来查看数据类型的函数

'''
    布尔类型
    对TRUE 不为零   错FALSE 0
'''
# a = True
# print(type(a))

'''
    字符串  string
    单引号或者双引号
'''
# a = '刘德华'
# name = "黎明"
# print(type(a))
# print(type(name))

# num = '1'
# num2 = 'True'  #带上引号的全部都是字符串
# print(type(num))
# print(type(num2))

# str1 = '"nihao"' #双引号成了字符串本身
# str1 = "'nihao'" #单引号成了字符串本身

# str1 = 'aaa\'bbbb'   #最外面的单引号是一对  里面的单引号通过 \ 转义
# print(str1)

# str = 'abcdfeg'  #从0开始 下标
# print(str[2:6])
# print(str[-2:-6:-1])

# print(str)
# print(str[0])
# print(str[1])
# print(str[2])

#截取字符串
# print(str[1:5])  #
# print(str[-2])  #
# print(str[2:])
# print(str[:2])

# print(str[1:5:2]) #步长
# print(str[5:2:-1]) #步长

#
num1 = '1'
num2 = 2
print(num1*num2)


# list01 = ['a','b',1,2]  #下标默认从 0 开始
# print(type(list01))
# print(list01[1])

num = [1,2,5,7,8]
#怎么两两作差
# num2 = num[2] - num[1]
# print(num2)
# list02 =[num[1] - num[0],num[2] - num[1],num[3] - num[2]]
# print(list02)


#元组
# tup01 = ('a','b',1,2)
# print(type(tup01))

# tup02 = (1,)
# print(type(tup02))

# tup03 = (213,23,5436,87,23,1)
# print(tup03[2])
# print(tup03[1:5])


#字典
# dict01 = {'name':'joe','age':19,'address':'上海'}
# print(dict01['name'])





'''
    运算符
'''
# +
# num1 = 10
# num2 = 3
# print(num1+num2)
# print(num1-num2)
# print(num1*num2)
# print(num1/num2)
# print(num1%num2)
# print(num1//num2)
# print(num1**num2)


'''
    赋值运算
'''
# str = 'a'

# num = 10
# num +=1   #num = num+1
# num -=1     #num = num - 1
# num *= 2    #num = num*2
# num /= 2    #num = num/2
# print(num)


'''
    比较运算符
'''
# a = 10
# b = 5
# print(a == b)
# print(a != b)
# print(a > b)
# print(a < b)

'''
    逻辑运算符
'''
# a = 10
# b = 5
# print(a>b and a<b)  #两个条件为真 则为真 否则返回假
# print(a>b or a<b )  #两个条件有一个为真则返回真，
# print( not a>b)


# str = ''   #空字符串 返回 bool false
# str = []   #孔的列表 返回 bool false
# str = ()     #空的元组 返回 bool false
# str = {}     #空的字典返回 bool false
# num = 0      #0 返回 bool false
# print(bool(num))

'''
    位运算
'''
a = 4
b = 2

'''
十进制: 1,2,3,4,5,6,7,8,9,10,11,12....19,20 
二进制:0000,0001,0010,0011,0100 
    0000 0100  4
    0000 0010  2
    0000 0110
  & 0000 0000  0

'''

# print(a & b)
# print(a | b)
# print(a ^ b)
#   # -4-1 -5
# print(~a)