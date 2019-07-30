# -*- coding:utf-8 -*-

"""
题目1：用一个数组实现两个栈，只需处理整型，实现l_pop/l_push/r_pop/r_push
"""


class two_stack:

    def __init__(self):
        self.data = []
        self.l_length = 0
        self.r_length = 0

    def l_pop(self):
        if self.l_length >= 1:
            self.l_length -= 1
            return self.data.pop(0)
        else:
            print("第一个栈内没有数据了!")
            return None

    def l_push(self, d):
        self.l_length += 1
        self.data.insert(0, d)

    def r_pop(self):
        if self.r_length >= 1:
            self.r_length -= 1
            return self.data.pop(-1)
        else:
            print("第二个栈内没有数据了!")
            return None

    def r_push(self, d):
        self.r_length += 1
        self.data.append(d)


if __name__ == '__main__':
    two_stack = two_stack()
    two_stack.l_push(1)
    two_stack.l_push(2)
    two_stack.r_push(3)
    two_stack.r_push(4)
    print(two_stack.l_pop())
    print(two_stack.l_pop())
    print(two_stack.r_pop())
    print(two_stack.r_pop())
    print(two_stack.r_pop())
