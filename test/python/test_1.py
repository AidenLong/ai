# -*- coding:utf-8 -*-

# 题目：现在有 a 到 z 26 个元素， 编写程序打印 a 到 z 中任取 3 个元素的组合（比如 打印 a b c ，d y z等）

def choice(n):
    for i in range(24):
        for j in range(i + 1, 25):
            for k in range(j + 1, 26):
                print(chr(ord('a') + i), end=' ')
                print(chr(ord('a') + j), end=' ')
                print(chr(ord('a') + k))


def bit(x):
    c = 0
    while x:
        c += 1
        x = (x & (x - 1))


def print_(x, count):
    i = 0
    if bit(x) == count:
        for i in range(26):
            if x & 1:
                print('%c', chr(ord('a') + i))
            x = (x >> 1)

if __name__ == '__main__':
    choice(2)
