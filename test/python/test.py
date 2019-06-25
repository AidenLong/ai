# -*- coding:utf-8 -*-
def f(n):
    if n == 1:
        return 1
    else:
        return n * f(n - 1)


# print(f(10000))

from functools import reduce

a = range(100)
print(reduce(lambda a, b: a + b, filter(lambda t: t % 2 == 0, a[100:])))
