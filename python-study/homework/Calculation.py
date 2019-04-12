# -*- coding:utf-8 -*-

class Calculation:

    def __init__(self, d1, d2):
        self.d1 = d1
        self.d2 = d2

    def add(self):
        print('输入数字的相加为：', self.d1 + self.d2)

    def sub(self):
        print('输入数字的相减为：', self.d1 - self.d2)

    def multi(self):
        print('输入数字的相乘为：', self.d1 * self.d2)

    def div(self):
        print('输入数字的相除为：', self.d1 / self.d2)

calculation = Calculation(6, 3)
calculation.add()
calculation.sub()
calculation.multi()
calculation.div()
