# -*- coding:utf-8 -*-

dict = {'1':['a', 'b' ,'c'], '2':['A', 'B', 'C']}
name = input("请输入名字:")
address = dict.get(name)
print(address)