# -*- coding:utf-8 -*-

# 将阶乘改成函数形式调用
# def factorial(n):
#     if n == 1:
#         return 1
#     else:
#         return n * factorial(n - 1)
#
#
# print(factorial(5))


# 写一个函数，求一个字符串的长度，在main函数中输入字符串，并输出其长度
# def strLeng(str):
#     return len(str)
#
#
# if __name__ == '__main__':
#     str = input("请输入一个字符串：")
#     print(strLeng(str))


# 有1,2,3,4个数字，能组成多少个互不相同且无重复数字的三位数？都是多少？
# # 删除list中的一个元素，返回新的list
# def delete(li, e):
#     index = li.index(e)
#     return li[:index] + li[index + 1:]
#
#
# def randomNum():
#     sum = 0
#     list1 = [1, 2, 3, 4]
#     for i in list1:
#         list2 = delete(list1, i)
#         for j in list2:
#             list3 = delete(list2, j)
#             for k in list3:
#                 print(i * 100 + j * 10 + k)
#                 sum += 1
#     print('总共%d个' % sum)
#
#
# randomNum()

# profit = int(input('请输入当月的利润:'))
# bonus = 0
# if profit <= 100000:
#     bonus = profit * 0.1
# elif 100000 < profit <= 200000:
#     bonus = 100000 * 0.1 + (profit - 100000) * 0.075
# elif 200000 < profit <= 400000:
#     bonus = 100000 * 0.1 + 100000 * 0.075 + (profit - 200000) * 0.05
# elif 400000 < profit <= 600000:
#     bonus = 100000 * 0.1 + 100000 * 0.075 + 200000 * 0.05 + (profit - 400000) * 0.03
# elif 600000 < profit <= 1000000:
#     bonus = 100000 * 0.1 + 100000 * 0.075 + 200000 * 0.05 + 200000 * 0.03 + (profit - 600000) * 0.015
# else:
#     bonus = 100000 * 0.1 + 100000 * 0.075 + 200000 * 0.05 + 200000 * 0.03 + 400000 * 0.015 + (
#                 profit - 1000000) * 0.01
#
# print('发放奖金总数为:%d' % bonus)

# for i in range(100, 1000):
#     a1 = i // 100
#     a2 = i % 100 // 10
#     a3 = i % 100 % 10
#     if a1 ** 3 + a2 ** 3 + a3 ** 3 == i:
#         print(i)

# import math
#
# for i in range(1000):
#     x = math.sqrt(i + 100)
#     # math.floor返回数字的下舍整数。若相等表示x为整数
#     if x == math.floor(x):
#         y = math.sqrt(x**2 + 168)
#         if y == math.floor(y):
#             print(i)

# num = int(input('请输入一个正整数：'))
#
#
# for i in range(2, num + 1):
#     flag = False
#     while True:
#         # 若果这个质数等于num，打印i，结束
#         if num == i:
#             print(i, end=' ')
#             flag = True
#             break
#         # 不等时，若能被整除，打印i，修改num，针对相同的i重新判断一次
#         if num % i == 0:
#             print(i, end=' ')
#             num = num / i
#         # 不能整除，跳出while循环，使用i+1判断
#         else:
#             break
#     if flag:
#         break


# def insertData(list1, e):
#     # 当添加的元素比list1中第一个或者最后一个元素大或者小时
#     lastIndex = len(list1) - 1
#     if list1[0] > list1[lastIndex]:
#         if e > list1[0]:
#             list1.insert(0, e)
#         elif e < list1[lastIndex]:
#             list1.append(e)
#     elif list1[0] < list1[lastIndex]:
#         if e < list1[0]:
#             list1.insert(0, e)
#         elif e > list1[lastIndex]:
#             list1.append(e)
#
#     # 添加元素在中间位置时
#     for i in range(0, lastIndex):
#         # 升序时
#         if list1[i] < e < list1[i + 1]:
#             list1.insert(i + 1, e)
#             break
#         # 降序时
#         elif list1[i] > e > list1[i + 1]:
#             list1.insert(i + 1, e)
#             break


# list1 = [1, 2, 5, 6]
# insertData(list1, 4)
# print(list1)
# list2 = [9, 7, 2, 1]
# insertData(list2, 10)
# print(list2)

# def add(n, a):
#     str1 = ''
#     for i in range(1, int(n) + 1):
#         str1 += a * i
#         str1 += '+'
#         print(a * i)
#     print(eval(str1[0:len(str1)-1]))
#
#
# n = input()
# a = input()
# add(n, a)


for n in range(1, 1000):
    l = []
    # 求出所有因子
    for a in range(1, int(n / 2 + 1)):
        if n % a == 0:
            l.append(a)

    # 判断是否是完数
    if sum(l) == n:
        print(n)
