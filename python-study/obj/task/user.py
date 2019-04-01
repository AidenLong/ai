#-*- conding:utf-8 -*-
''''''
'''
3.用户：创建一个名为UserUser的类，其中包含属性first_name和last_name,
还有用户简介通常会存储的其他几个属性。在类User中定义一个名为
describe_user()的方法，它打印用户信息摘要；
再定义一个名为greet_user()的方法，它向用户发出个性化的问候。
创建多个表示不同用户的实例，并对每个实例都调用上述两个方法.

'''
class User():
    '''
        用户类
    '''
    def __init__(self,first_name,last_name,age,sex,phone,login_attempts=0):
        #初始化 实例属性
        self.first_name = first_name
        self.last_name = last_name
        self.age = age
        self.sex = sex
        self.phone = phone
        self.login_attempts = login_attempts

    #查看用户信息
    def describe_user(self):
        print('大家好我叫%s %s,我今年%d岁，我的电话是%s'%(self.first_name,self.last_name,self.age,self.phone))


    #个性化问候
    def greet_user(self):
        print('尊敬的%s，恭喜你中了五百万。'%self.first_name)

    #增加登录次数
    def increment_login_attempts(self):
        self.login_attempts += 1
        print('当前登录次数%d'%self.login_attempts)
    #重置登录次数
    def reset_login_attempts(self):
        self.login_attempts = 0
        print('当前登录次数%d' % self.login_attempts)


if __name__ == '__main__':

    joe = User('joe','black',19,'男','18600009999')
    joe.describe_user()
    joe.greet_user()

    joe.increment_login_attempts()
    joe.increment_login_attempts()
    joe.increment_login_attempts()

    joe.reset_login_attempts()


