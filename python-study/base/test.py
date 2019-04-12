# -*- coding:utf-8 -*-

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

