# -*- coding:utf-8 -*-
from obj.task.Restaurant import Restaurant

'''
冰淇淋小店：冰其淋小店是一种特殊的餐馆。编写一个名为IceCreamStand的类，
让它继承你为完成练习1或练习4而编写的Restaurant类。这两个版本Restaurant类都可以，
挑选你更喜欢的那个即可。添加一个名为flavors的属性,用于存储一个由各种口味的冰淇淋组成的列表。
编写一个显示这些冰淇淋的方法。创建一IceCreamStand实例,并调用这个方法。
'''
class IceCreamStand(Restaurant):
    def __init__(self, restaurant_name, cuisine_type, flavors):
        super(IceCreamStand, self).__init__(restaurant_name, cuisine_type)
        self.flavors = flavors

    def show(self):
        print('出售的味道有：',end=" ")
        for i in self.flavors:
            print(i,end=",")

if __name__ == '__main__':
    iceCreamStand = IceCreamStand('学麒麟', '冰淇淋', ['草莓','原味'])
    iceCreamStand.describe_restaurant()
    iceCreamStand.show()