# -*- coding:utf-8 -*-

from functools import reduce

a = [1, 2, 3, 4, 5, 6, 7]
b = [i for i in filter(lambda x: x > 5, a)]
print(b)

a = range(100)
print(reduce(lambda a, b: a + b, filter(lambda t: t % 2 == 2, a[100:])))
