# -*- coding:utf-8 -*-

# 使用for循环实现1-100的和
# sum = 0
# for i in range(101):
#     sum += i
# print(sum)

# 99乘法表
# for i in range(1, 10):
#     for j in range(1, i + 1):
#         print(j, '*' , i, "=", i*j, end='\t')
#     print()

# 从键盘输入一个字符串，将小写字母全部转换成大写字母,将字符串以列表的形式输出(如果字符串包含整数取整型)?
# str = input("输入一个字符串:")
# list1 = []
# for i in str:
#     if i.isdigit():
#         list1.append(int(i))
#     else:
#         list1.append(i.upper())
# print(list1)

# 随机输入8位以内的的正整数，要求：一、求它是几位数，二、逆序打印出各位数字。
num = input("输入一个正整数:")
length = len(num)
if length <= 8:
    print(length)
    for i in num[::-1]:
        print(i, end=" ")


# 一球从n米(自己输入)高度自由落下，每次落地后反跳回原高度的一半；再落下，求它在第10次落地时，共经过多少米？第10次反弹多高？
# heigh = int(input("输入一个正整数:"))
# i = 0
# sum = 0
# while i <= 10 :
#     sum += heigh
#     heigh /= 2
#     i += 1
#
# print(sum)
# print(heigh)

# 输入一行字符，分别统计出其中英文字母、空格、数字和其它字符的个数
# str = input("输入一个正字符:")
# a = 0
# space = 0
# num = 0
# other = 0
# for i in str:
#     if i.isdigit():
#         num += 1
#     elif i.isalpha():
#         a += 1
#     elif i == " ":
#         space += 1
#     else:
#         other += 1
#
# print("英文字母%d、空格%d、数字%d和其它字符%d" % (a, space, num, other))
