# -*- coding:utf-8 -*-
# 既不需要侵入，也不需要函数重复执行
import time

'''
    python装饰器就是用于拓展原来函数功能的一种函数，
    这个函数的特殊之处在于它的返回值也是一个函数，
    使用python装饰器的好处就是在不用更改原函数的代码前提下给函数增加新的功能。 
'''
def deco(func):
    def wrapper():
        startTime = time.time()
        func()
        endTime = time.time()
        msecs = (endTime - startTime) * 1000
        print("time is %d ms" % msecs)

    return wrapper


@deco
def func():
    print("hello")
    time.sleep(1)
    print("world")


if __name__ == '__main__':
    f = func  # 这里f被赋值为func，执行f()就是执行func()
    f()
