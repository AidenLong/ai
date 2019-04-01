# -*- conding:utf-8 -*-
class Animal():
    def __init__(self, name, food):
        self.name = name
        self.food = food

    def eat(self):
        print('%s爱吃%s' % (self.name, self.food))


# 声明一个子类继承animal
class Dog(Animal):
    def __init__(self, name, food, drink):
        # 加载弗雷构造方法
        # Animal.__init__(self,name,food,)
        super(Dog, self).__init__(name, food)
        self.drink = drink  # 子类自己的属性

    # 子类自己的方法
    def drinks(self):
        print('%s爱喝%s' % (self.name, self.drink))


class Cat(Animal):
    def __init__(self, name, food, drink):
        # 加载弗雷构造方法
        # Animal.__init__(self,name,food,)
        super(Cat, self).__init__(name, food)
        self.drink = drink  # 子类自己的属性

    # 子类自己的方法
    def drinks(self):
        print('%s爱喝%s' % (self.name, self.drink))

    # 重写父类的eat
    def eat(self):
        print('%s特别爱吃%s' % (self.name, self.food))


dog1 = Dog('金毛', '骨头', '可乐')
dog1.eat()
dog1.drinks()

cat1 = Cat('波斯猫', '秋刀鱼', '雪碧')
cat1.eat()
cat1.drinks()
