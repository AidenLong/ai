# -*- coding:utf-8 -*-

class MinStack(object):

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.data = []

    def push(self, x):
        """
        :type x: int
        :rtype: None
        """
        self.data.append(x)

    def pop(self):
        """
        :rtype: None
        """
        return self.data.pop()

    def top(self):
        """
        :rtype: int
        """
        return self.data[-1]

    def getMin(self):
        """
        :rtype: int
        """
        return min(self.data)


if __name__ == '__main__':
    minStack = MinStack()
    minStack.push(-2)
    minStack.push(0)
    minStack.push(-3)
    print(minStack.getMin())  # --> 返回 - 3.
    print(minStack.pop())
    print(minStack.top())  # --> 返回 0.
    print(minStack.getMin())  # --> 返回 - 2.
