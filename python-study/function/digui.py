# -*- conding:utf-8 -*-
'''
    递归函数
'''
# def main(n):
#     print('进入第%d层梦境'%n)
#     if n == 3:
#         print('到达潜意识区,原来我最爱的人是你！开始醒来')
#     else:
#         main(n+1)
#     print('从第%d层梦境醒来'%n)
#
#
# main(1)  #回到调用的地方
'''
    第一次调用:进入第1层梦境
        第二次调用：进入第2层梦境
            第三次调用：进入第3层梦境 进入 if 到达潜意识区,原来我最爱的人是你！开始醒来
            从第3层梦境醒来 结束第三次调用
        从第2层梦境醒来 结束第二次调用
    从第1层梦境醒来 结束第一次调用
'''

'''
    阶乘
'''
# def jiecen(n):
#     if n == 1:
#         return 1
#     else:
#         return n*jiecen(n-1)
#
# print(jiecen(5))

'''
斐波那契数列（Fibonacci sequence）指的是这样一个数列：1、1、2、3、5、8、13、21、34、……在数学上，
斐波纳契数列以如下被以递归的方法定义：F(0)=0，F(1)=1, F(n)=F(n-1)+F(n-2)（n>=2，n∈N*）
'''
def fib(n):
    if n <= 1:
        return n
    else:
        return fib(n - 1) + fib(n - 2)

print(fib(8))

'''
    匿名函数
    lambda def [参数](): 表达式 return xx
    默认会返回一个表达式
'''
# 没有参数的lambda表达式
# s = lambda : '哈哈啊哈'   #通过；lambda 声明一个匿名函数 并且 赋值给 s
# print(s())   #通过s调用匿名函数

# 声明一个参数
# s = lambda x: x*2
# print(s(3))

# s = lambda x,y: x+y
# print(s(3,4))

'''
    矢量化的三元运算符
    if  条件：
        代码块哦
    else：
        代码块

    条件成立的内容   if 条件 else 条件不成立的内容
'''
# s = lambda x, y: x if x > 2 else y
# print(s(1, 4))
