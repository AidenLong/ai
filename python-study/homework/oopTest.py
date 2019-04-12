# -*- coding:utf-8 -*-

class Bank:
    __users = {}

    def __init__(self, name, id, Balance):
        if id not in Bank.__users:
            self.__name = name
            self.__id = id
            self.__Balance = Balance
            Bank.__users[id] = {'name': name, 'Balance': Balance}
        else:
            print('此id %d 的用户已存在' % id)

    @classmethod
    def get_Users(cls):
        for key, val in cls.__users.items():
            print('卡号：%s \n 用户名：%s \n 余额：%d' % (key, val['name'], val['Balance']))
            print()

    # 验证金额
    @staticmethod
    def check_money(money):
        if isinstance(money, int):
            return True
        else:
            return False

    def save_money(self, id, money):
        if id not in Bank.__users.keys():
            print('输入卡号有误')
        else:
            # 开始存钱
            if Bank.check_money(money):
                Bank.__users[id]['Balance'] += money
                print('当前卡号%s,当前存款金额%d,当前余额%d' % (id, money, Bank.__users[id]['Balance']))
            else:
                print('您输入的金额有误')

    def consume_money(self, id, money):
        if id not in Bank.__users.keys():
            print('输入卡号有误')
        else:
            # 开始取钱
            if Bank.check_money(money):
                if Bank.__users[id]['Balance'] >= money:
                    Bank.__users[id]['Balance'] -= money
                    print('当前卡号%s,当前取款金额%d,当前余额%d' % (id, money, Bank.__users[id]['Balance']))
                else:
                    print('余额不足')
            else:
                print('您输入的金额有误')

    # 查看个人详细信息
    def getInfo(self, id):
        if id not in Bank.__users.keys():
            print('输入卡号有误')
        else:
            print('当前卡号%s,当前用户名%s,当前余额%d' % (id, Bank.__users[id]['name'], Bank.__users[id]['Balance']))


joe = Bank('joe', 101, 100)
jos = Bank('jos', 102, 100)
Bank.get_Users()
print('_' * 50)
joe.save_money(101, 500)
print('_' * 50)
joe.consume_money(101, 300)
print('_' * 50)
joe.getInfo(101)
