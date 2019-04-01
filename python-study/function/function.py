#-*- conding:utf-8 -*-
'''
'''
'''
  什么是函数？
    -》组织好的，可重复使用的，用户实现单一或者关联功能的代码段
  函数的作用？
    -》提高应用的模块性，和代码的重复使用率 
'''
'''
    函数的定义：
    def 函数名([参数]):
        #函数说明
        要封装的代码块
'''

# def Pname():   #当前函数不放参数
#     '''
#         获取姓名
#     :return:
#     '''
#     print('大家好我是小明同学！')

#有参数吗？  很明显没有
# Pname()  #调用函数  执行了函数里面的代码
# pri = Pname   #将函数名赋值给另一个变量  ，给当前函数取一个别名
# pri()

# def getNum():   #定义函数
#     print('100')
# getNum()#调用函数

'''
    函数的参数
'''
# def Pname(userName):   #userName 形参  形参的名字是自定义的
#     '''
#         获取姓名
#     :return:
#     '''
#     print('大家好我是%s同学！'%userName)
#
# Pname('帅哥')  #传递了一个实参  '刘德华'


#必备参数
# def getInfo(name,address):
#     print('大家好我叫%s，我来自%s'%(name,address))

#getInfo('刘德华','香港')  #第一个实参对应了第一个形参，第二。。。对应第二个。。
# getInfo('香港','刘德华')  #第一个实参对应了第一个形参，第二。。。对应第二个。。
# getInfo('刘德华') #形参有两个，调用的时候，实参也要传递两个，否则会报错
#参数的个数不能少，不能多。 参数的位置要一一对应


#关键字参数
# def getInfo(name,address):
#     print('大家好我叫%s，我来自%s'%(name,address))
#
# getInfo(name='刘德华',address='香港')  #给实参加上关键字 。关键字对应我们的形参

#参数的默认值
# def getInfo(name,address = '香港'): #默认值参数，就是在声明函数的时候给形参赋值
#     print('大家好我叫%s，我来自%s'%(name,address))

# getInfo('刘德华')   #有默认值的形参，可以不用传递
# getInfo('刘德华','九龙')   #传递参数的花，会覆盖原来的默认值



#不定长参数
# def getInfo(name,address,*args,**agrs2):
#     print('大家好我叫%s，我来自%s'%(name,address))
#     print(args)  #args 是一个元组类型
#     print(agrs2)  #字典数据类型
#
# getInfo('刘德华','九龙','a','b','c','d',age = 18)

#*args 是接受所有未命名的参数（关键字）
#**agrs2 是接受所有命名的参数（带关键字的）

'''
    可变对象与不可变对象的传递
'''
#标记函数
def sign():  #自定义的函数
    print('_'*50)



#值传递  ： 不可变对象的传递
# def fun(args):
#     args = 'hello'   #重新赋值
#     print(args)   #输出hello
#
# str1 = 'baby' #声明一个字符串的变量   不可变数据类型
# fun(str1)    #将该字符串传递到函数中
# sign()
# print(str1)   #还是baby  并没有被改变

#引用传递  ： 可变对象的传递
# def fun(args):
#     args[0] = 'hello'   #重新赋值
#     print(args)   #输出
#
# list01 = ['baby','come on'] #声明一个列表，可变数据类型
# fun(list01)    #将该列表传递到函数中
# sign()
# print(list01)  #传递的是对象本身，函数里面被修改了值，原对象也会跟着修改


