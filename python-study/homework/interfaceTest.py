# -*- coding:utf-8 -*-
import math

class Shape:

    def girth(self):
        pass

    def area(self):
        pass


class Circle(Shape):

    def __init__(self, r):
        self.r = r

    def girth(self):
        print('周长等于:', 2 * math.pi * self.r)

    def area(self):
        print('面积等于:', math.pi * math.pow(self.r, 2))


class Rectangle(Shape):

    def __init__(self, width, high):
        self.width = width
        self.high = high

    def girth(self):
        print('周长等于:', 2 * (self.width + self.high))

    def area(self):
        print('面积等于:', self.width * self.high)

circle = Circle(3)
circle.girth()
circle.area()
rectangle = Rectangle(2, 4)
rectangle.girth()
rectangle.area()
