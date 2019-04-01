#-*- conding:utf-8 -*-
''''''
'''
    函数的返回值
    return关键字实现
'''
def max(x,y):
    if x >y:
        return x  #结束函数的运行 并且将结果返回给调用的地方
    else:
        return y
        # print(y)  #没有执行print
        #后面的代码不会执行
#调用函数  接受返回值
# num = max(1,2)  #声明一个变量num 接受调用函数后的返回值
# print(max(1,2))  #观察接受的返回值   返回值会返回到调用的地方


#return返回多个返回值
# def sum(x,y):
#     return x,y
#
# num = sum(1,2)  #用一个变量接受多个返回值，会保存在一个元组中
# print(num)

# num1,num2 = sum(1,2)
# print(num1)
# print(num2)


'''
    return 有什么用？
    在哪里获取返回值？  将结果返回到调用的地方
    return 之后下面的代码会运行吗？
'''


'''
    yield 生成器
    生成一个迭代器
        -》yield的作用是吧一个函数变成一个generator
        -》使用生成器可以达到延迟操作的效果，所谓延迟操作就是指在需要的时候
        产生结果而不是立即产生就结果，节省资源消耗，和声明一个序列不同的是
        生成器，在不使用的时候几乎是不占内存的。
'''

# def getNum(n):
#     i = 0
#     while i <= n:
#         #print(i)    #打印i
#         #return i     #返回一个i ,结束函数的运行
#         yield i     #将函数变成一个generator
#         i+=1
# # 调用函数
# print(getNum(5))
#
# a = getNum(5)  #把生成器赋值给一个变量a
# # 使用生成器 通过 next()方法
# print(next(a))  #输出yield返回的值
# print(next(a))
# print(next(a))
# print(next(a))
# print(next(a))
# print(next(a))

# print(next(a))

#for循环遍历一个生成器
# for i in a:
#     print(i)

# a = [x for x in range(10000000)]   #这样生成一个很多数据的列表会占用很大的内存
# print(a)
# a = (x for x in range(10000000))  #不是元组推导式
# print(a)
# print(next(a))
# print(next(a))
# print(next(a))


'''
    send
'''
# def gen():
#     i = 0
#     while i < 5:
#         temp = yield i  #是赋值操作吗？不是
#         #使用了yield之后是一个生成器
#         print(temp)   #因为 yield 之后返回结果到调用者的地方，暂停运行 ，赋值操作没有运行
#         i+=1
# a = gen()
# print(next(a))
# print(next(a))
# print(a.send('我是a'))  #可以将值发送到 上一次yield的地方


'''
    迭代器
    什么是迭代对象？
    可以用for in 遍历得对象度可以叫做是可迭代对象:Iterable
    list string dict  
    
    可以被next()函数调用的并不断返回下一个值得对象叫做迭代器：iterator
        凡是可以用作与next()函数的对象都是iterator
'''

list01 = [1,2,3,4,5] #是一个可迭代对象
# for i in list01:
#     print(i)
# print(next(list01))   #list01不是迭代器所以无法调用  next

#通过iter()将一个可迭代对象变成迭代器
# a = iter(list01)
# print(a)
# print(next(a))
# print(next(a))
# print(next(a))



















