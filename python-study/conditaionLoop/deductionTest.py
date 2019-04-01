# -*- coding:utf-8 -*-
names = [['Tom', 'Billy', 'Jefferson', 'Andrew', 'Wesley', 'Steven', 'Joe'],
         ['Alice', 'Jill', 'Ana', 'Wendy', 'Jennifer', 'Sherry', 'Eva']]
# 找出上述名字中长度大于4的名字,组成列表打印出来.
# 过滤掉长度大于5的字符串列表，并将剩下的转换成大写字母.

# names1 = [n for i in names for n in i if len(n) > 4]
# print(names1)
# names1 = [n.upper() for i in names for n in i if len(n) < 5]
# print(names1)

# 求M,N中矩阵对应元素的和，元素的乘积
# m = [[1, 2, 3],
#      [4, 5, 6],
#      [7, 8, 9]]
# n = [[2, 2, 2],
#      [3, 3, 3],
#      [4, 4, 4]]
#
# result = [[m[i][j] + n[i][j] for j in range(len(m[0]))] for i in range(len(m))]
# print(result)
# result = [[m[i][j] * n[i][j] for j in range(len(m[0]))] for i in range(len(m))]
# print(result)

'''
打印出所有的“水仙花数”，所谓“水仙花数”是指一个三位数，其各位数字立方和等于该数本身。
     例如：153是一个“水仙花数”，因为153=1的三次方＋5的三次方＋3的三次方。
程序分析：利用for循环控制100-999个数，每个数分解出个位，十位，百位。
'''

# for i in range(100, 1000):
#     sum = 0
#     for j in str(i):
#         sum += int(j) ** 3
#     if sum == i:
#         print(i)
#
# for i in range(100, 1000):
#     a1 = i // 100
#     a2 = i % 100 // 10
#     a3 = i % 100 % 10
#     if a1 ** 3 + a2 ** 3 + a3 ** 3 == i:
#         print(i)

'''
    打印菱形
'''
# for i in range(1, 8, 2):
#     print(("*" * i).center(7))
#     if i == 7:
#         for i in range(5, 0, -2):
#             print(("*" * i).center(7))

'''
一个5位数，判断它是不是回文数。即12321是回文数，个位与万位相同，十位与千位相同
'''
# num = str(4123214)
# flag = True
# for i in range(len(num) // 2):
#     if num[i] != num[-(i + 1)]:
#         flag = False
#         break
#
# if flag:
#     print(num)
#
# print(num == num[::-1])


'''
求一个3*3矩阵对角线元素之和 
'''
# list1 = [[4, 1, 4],
#          [2, 4, 2],
#          [4, 3, 4]]
#
# x = 0
# y = 0
# for i in range(len(list1)):
#     x += list1[i][i]
#     y += list1[i][len(list1) - 1 - i]
#
# print(x)
# print(y)

'''
题目：有四个数字：1、2、3、4，能组成多少个互不相同且无重复数字的三位数？各是多少？ 
程序分析：可填在百位、十位、个位的数字都是1、2、3、4。
组成所有的排列后再去 掉不满足条件的排列。(用列表推导式)
'''
# for i in range(1, 5):
#     for j in range(1, 5):
#         for k in range(1, 5):
#             if i != j and i != k and j != k:
#                 print(i * 100 + j * 10 + k)

list1 = [i * 100 + j * 10 + k for i in range(1, 5) for j in range(1, 5) for k in range(1, 5) if
         i != j and i != k and j != k]
print(list1)
'''
将列表用for循环添加到另一个字典中.
'''
# names = ['Tom', 'Billy', 'Jefferson', 'Andrew', 'Wesley', 'Steven', 'Joe',
# #          'Alice', 'Jill', 'Ana', 'Wendy', 'Jennifer', 'Sherry', 'Eva']
# # name = {k:v for k, v in enumerate(names)}
# # print(name)

'''
设一组账号和密码不少于两个
通过密码和账号，如果输入正确显示登陆成功
若账号或密码错误则显示密码或者账号错误，最多输错三次
'''
# users = {"张三":'123456', '李四':'654321'}
# num = 0
# while num < 3:
#     name = input('请输入账号：')
#     password = input('请输入密码：')
#     if name not in users:
#         num += 1
#         print('密码或者账号错误%d次' % num)
#     elif users[name] == password:
#         print('登陆成功')
#         break
#     else:
#         num += 1
#         print('密码或者账号错误%d次'%num)


'''
求阶乘，用while和for实现
如：2的阶乘 2*1    3的阶乘 4*3*2*1
'''
# n = int(input("请输入一个整数："))
# result = 1
# # for i in range(2, n + 1):
# #     result *= i
#
# i = 1
# while i <= n:
#     result *= i
#     i += 1
#
# print(result)


'''
冒泡排序
'''
list01 = [2, 6, 4, 3, 1, 9]
for i in range(len(list01)):
    for j in range(1, len(list01) - i):
        # print(list01[j], end=" ")
        # print(list01[j - 1], end=" ")
        if list01[j] > list01[j - 1]:
            list01[j], list01[j - 1] = list01[j - 1], list01[j]
    # print("-"*20)
print(list01)
