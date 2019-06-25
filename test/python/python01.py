# -*- coding:utf-8 -*-
def f(n):
    pre = 1
    sum = 1
    if n < 1:
        return -1
    elif n == 1:
        return sum
    else:
        for i in range(2, n + 1):
            pre = i * pre
            sum += pre
        return sum


print(f(10000))
