#-*- conding:utf-8 -*-
''''''
'''
    import 模块名[,模块名]。。。
    导入整个模块
    通过 模块名.方法名 调用
'''
# import time
# print('start')
# # sleep(5)  错误调用
# time.sleep(5)  #正确调用
# print('stop')、

'''
    from 模块名 import 函数名
    从指定的模块中导入指定的部分
'''
# from time import sleep
# print('start')
# sleep(5)
# print('stop')


#导入模块中的所有内容
# from math import *
# print(ceil(1.1))  #向上取整
# print(floor(1.1)) #向下取整


#给导入的模块取别名
# import math as m
# print(m.ceil(1.1))  #向上取整
# print(m.floor(1.1)) #向下取整

# from math import ceil as c  #不建议，函数名组好简化或者区别名
# print(c(1.1))  #向上取整
import math
print(math.floor(1.1))


