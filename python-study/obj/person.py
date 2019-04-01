# -*- coding:utf-8 -*-
'''
    类定义
    class 类名():
        # 类文档说明
        属性
        方法
'''
class person():
    '''
        这是一个人类
    '''
    country = '中国'  #声明类属性，并且赋值
    #实例属性通过构造方法来声明
    #self不是关键字，代表的是当前而对象
    def __init__(self,name,age,sex): #构造方法
        #构造方法不需要调用，在实例化的时候自动调用
        # print('我是构造方法，在实例化得时候调用')
        self.name = name  #通过self 创建实例属性，并且赋值
        self.age = age
        self.sex = sex

    #创建普通方法
    def getName(self):
        print('我的名字叫：%s,我来自%s'%(self.name,person.country)) #在方法里面使用实例属性

#实例化对象
people01 = person('joe',19,'男')  #在实例化的时候传递参数
#这个people01 就要具有三个属性，并且可以使用getName方法


#访问属性
# print(people01.name)  #通过对象名.属性名 访问实例属性(对象属性)
# print(people01.age)
# print(people01.sex)

#通过内置方法访问属性
# print(getattr(people01,'name'))
# print(hasattr(people01,'name'))
#
# setattr(people01,'name','susan')
# print(people01.name)
#
# delattr(people01,'name')
# print(people01.name)

#通过对象调用实例方法
people01.getName()

'''
    内置类属性
'''
print(people01.__dict__)  #会将实例对象的属性和值通过字典的形式返回
# print(people01.__doc__)
# print(person.__name__)   #返回类名
# print(person.__bases__)

people02 = person('susan',19,'女')
people02.getName()
