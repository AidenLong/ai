# -*- coding:utf-8 -*-

# 输入两个数，计算加减乘除
flag = True
while flag:
    try:
        a = int(input('输入第一个数:'))
        b = int(input('输入第二个数:'))
        print('输入数字的相加为：', a + b)
        print('输入数字的相减为：', a - b)
        print('输入数字的相乘为：', a * b)
        print('输入数字的相除为：', a / b)
        flag = False
    except ValueError:
        print('请输入数字！')

# 创建一个集合，存储一些数据遍历出来
list1 = ['小红', '小明']

for i in list1:
    print("%s 你好" % i)

# 创建一个字典，存储一些数据遍历出来
dict2 = {'姓名': '小明', '年龄': 18, '地址': '北京'}
for k, v in dict2.items():
    print(k, v)

# 接收输入的姓名和年龄，然后输出
name = input('输入你的姓名：')
age = input('输入你的年龄：')

print(name)
print(age)
