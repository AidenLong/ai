# -*- coding:utf-8 -*-

'''
    变量的作用域
'''
#局部变量：声明在函数内部的变量是局部变量
# def test1():
#     a = 1
#     print(a)
#     print(id(a))
#
# def study():
#     a = 2
#     print(a)  #无法使用test1函数中定义的局部变量
#     print(id(a))
# test1()
# study()
# print(a)  #局部变量的作用于只在于函数中，外部无法使用



# #全局变量
# def test1():
#     print(a)
#     print(id(a))
#
# def study():
#     print(a)  #无法使用test1函数中定义的局部变量
#     print(id(a))
#
# a = 1
# test1()
# study()
#
# print(a)
# print(id(a))  #大家使用的是同一个变量，去哪局变量


''''
    修改全局变量
'''
# def test1():
#     global a  #声明去哪局变量a
#     a = 2
#     print(a)
#     print(id(a))
#
# def study():
#     global a
#     a = 3
#     print(a)  #无法使用test1函数中定义的局部变量
#     print(id(a))
#
# a = 1
# test1()
# study()
# print(a)
# print(id(a))



'''
    练习：ATM机
'''

#登录验证
def login(passwd):
    pwd = '888888'
    if passwd == pwd:
        return True
    else:
        return False

#2.金额验证
def checkMoney(money):
    if money.isdigit():
        if int(money) % 100 ==0 and 0<= int(money) <=1000:
            return money
        else:
            return False
    else:
        return False

#业务逻辑写到主程序当中
def main():
    #1.登录验证
    for i in range(3):
        passwd = input('请输入密码:')
        if login(passwd):
            # 2.金额验证
            while True:
                money= input('请输入金额：')
                money = checkMoney(money)
                if money :
                    print('成功取出%s元'%money)
                    break
                else:
                    print('输入的金额有误！请重新输入')
            # 3.交易完成
            print('交易完成')
            break
        else:
            if i == 2:
                print('您连续三次密码有误，账号已冻结！')
                break
            print('密码有误')

main()



'''
    分数积
'''
# def EvenSum(n):
#     x = 0
#     for i in range(2,n+2,2):
#         x+= 1/i
#     return x
#
# def OddSum(n):
#     x = 0
#     for i in range(1,n+2,2):
#         x+= 1/i
#     return x
#
#
# def main():
#     #接受一个数字
#     num = int(input('请输入一个数字：'))
#     #判断奇偶
#     if num%2 == 0:
#         k = EvenSum(num)
#         print('偶数的分数积是:%f'%k)
#     else:
#         k = OddSum(num)
#         print('奇数的分数积是:%f'%k)
#
# main()




