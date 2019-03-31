# -*- coding:utf-8 -*-
names = [['Tom', 'Billy', 'Jefferson', 'Andrew', 'Wesley', 'Steven', 'Joe'],
         ['Alice', 'Jill', 'Ana', 'Wendy', 'Jennifer', 'Sherry', 'Eva']]
# 找出上述名字中长度大于4的名字,组成列表打印出来.
# 过滤掉长度大于5的字符串列表，并将剩下的转换成大写字母.

names1 = [n for i in names for n in i if len(n) > 4]
print(names1)
names1 = [n.upper() for i in names for n in i if len(n) < 5]
print(names1)

# 求M,N中矩阵对应元素的和，元素的乘积
m = [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]
n = [[2, 2, 2],
     [3, 3, 3],
     [4, 4, 4]]

result = [[m[i][j] + n[i][j] for j in range(len(m[0]))] for i in range(len(m))]
print(result)
result = [[m[i][j] * n[i][j] for j in range(len(m[0]))] for i in range(len(m))]
print(result)


'''
打印出所有的“水仙花数”，所谓“水仙花数”是指一个三位数，其各位数字立方和等于该数本身。
     例如：153是一个“水仙花数”，因为153=1的三次方＋5的三次方＋3的三次方。
程序分析：利用for循环控制100-999个数，每个数分解出个位，十位，百位。
'''

for i in range(100, 1000):
    sum = 0
    for j in str(i):
        sum += int(j) ** 3
    if sum == i:
        print(i)

'''
一个5位数，判断它是不是回文数。即12321是回文数，个位与万位相同，十位与千位相同
'''
num = str(4123214)
flag = True
for i in range(len(num) // 2):
    if num[i] != num[-(i + 1)]:
        flag = False
        break

if flag:
    print(num)


'''
求一个3*3矩阵对角线元素之和 
'''
list1 = [[4, 1, 4],
         [2, 4, 2],
         [4, 3, 4]]

print()
