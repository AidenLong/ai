#-*- conding:utf-8 -*-
''''''
# def fun_add(x,y):
#     return x+y
#
# print(fun_add(2,3))
# print('hahaha')


def f():
    print('start')
    a = yield 1  #表达式都是从右向左开始,首先返回1 然后a=1 yield 1  a=1
    print(a)
    print('middle....')
    b = yield 2  # 2这个值只是迭代值，调用next时候返回的值
    print(b)
    print('next')
    c = yield 3
    print(c)

a = f()
# print(next(a))
# print(next(a))
# print(next(a))
print(next(a))
# print(a.send(None))
print(a.send('msg'))
print(a.send('msg1'))
print(next(a))
print(next(a))
