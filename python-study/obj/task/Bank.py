# -*- coding:utf-8 -*-
'''
模拟一个简单的银行进行业务办理的类

类：
    创建一个银行类
属性：
    一个属于银行的类属性
        用来存储所用银行的开户信息，包含卡号、密码、用户名、余额
        （外界不能随意访问和修改。开户时要进行卡号验证，查看卡号是否已经存在）

    每个对象拥有
        卡号、密码、用户名、余额
        （外界不能随意访问和更改）

方法：
    银行类拥有
        查看本银行的开户总数
        查看所有用户的个人信息（包含卡号、密码、用户名、余额）
    每个对象拥有
        实例化对象的时候传入相关参数
            初始化对象及类属性
        取钱（需要卡号和密码验证）
            通过验证卡号和密码对个人的余额进行操作，如果取钱大于余额，返回余额不足
        存钱（需要卡号和密码验证）
            通过验证卡号和密码对个人的余额进行操作，返回操作成功
        查看个人详细信息（需要卡号密码验证）
            返回个人的卡号，用户名，余额信息
'''


class Bank():
    # 一个属于银行的类属性
    __Users = {}

    # 每个对象拥有 卡号、密码、用户名、余额
    def __init__(self, CradId, pwd, name, balance):
        if CradId not in Bank.__Users:
            Bank.__Users[CradId] = {'pwd': pwd, 'Username': name, 'Balance': balance}
            self.__CradId = CradId
            self.__pwd = pwd
            self.__name = name
            self.__balance = balance
        else:
            print('该用户已经存在')

    # 查看本银行的开户总数
    @classmethod
    def nums(cls):
        print('当前用户数：%d' % (len(cls.__Users)))

    # 查看所有用户的个人信息（包含卡号、密码、用户名、余额）
    @classmethod
    def get_Users(cls):
        for key, val in cls.__Users.items():
            print('卡号：%s \n 用户名：%s \n密码：%d \n 余额：%d' % (key, val['Username'], val['pwd'], val['Balance']))
            print()

    # 验证方法
    @staticmethod
    def check_User(CradId, pwd):
        if (CradId in Bank.__Users) and (pwd == Bank.__Users[CradId]['pwd']):
            return True
        else:
            return False

    # 验证金额
    @staticmethod
    def check_money(money):
        if isinstance(money, int):
            return True
        else:
            return False

    # 取钱（需要卡号和密码验证）
    def q_money(self, CradId, pwd, money):
        if Bank.check_User(CradId, pwd):
            # 开始取钱
            if Bank.check_money(money):
                if Bank.__Users[CradId]['Balance'] >= money:
                    Bank.__Users[CradId]['Balance'] -= money
                    print('当前卡号%s,当前取款金额%d,当前余额%d' % (CradId, money, Bank.__Users[CradId]['Balance']))
                else:
                    print('余额不足')
            else:
                print('您输入的金额有误')
        else:
            print('卡号或者密码有误')

    def c_money(self, CradId, pwd, money):
        if Bank.check_User(CradId, pwd):
            # 开始取钱
            if Bank.check_money(money):
                Bank.__Users[CradId]['Balance'] += money
                print('当前卡号%s,当前存款金额%d,当前余额%d' % (CradId, money, Bank.__Users[CradId]['Balance']))
            else:
                print('您输入的金额有误')
        else:
            print('卡号或者密码有误')

    # 查看个人详细信息（需要卡号密码验证）
    def getInfo(self, CradId, pwd):
        if Bank.check_User(CradId, pwd):
            print('当前卡号%s,当前用户名%s,当前余额%d' % (CradId, Bank.__Users[CradId]['Username'], Bank.__Users[CradId]['Balance']))
        else:
            print('卡号或者密码有误')


joe = Bank('1001', 111111, 'joe', 100)
jos = Bank('1002', 111111, 'jos', 100)
Bank.nums()
print('_' * 50)
Bank.get_Users()
print('_' * 50)
joe.c_money('1001', 111111, 500)
print('_' * 50)
joe.q_money('1001', 111111, 300)
print('_' * 50)
joe.getInfo('1001', 111111)
