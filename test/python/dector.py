# -*- coding:utf-8 -*-
def log(func):
    def inner(*args, **kwargs):
        print('start function {}'.format(func.__name__))
        func(*args, **kwargs)
        print('end function {}'.format(func.__name__))

    return inner


@log
def add(a, b):
    print('%d + %d = %d' % (a, b, a + b))


@log
def add2(a, b, c):
    print('%d + %d + %d = %d' % (a, b, c, a + b + c))


add(2, 4)
add2(2, 4, 6)
