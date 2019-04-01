# -*- coding:utf-8 -*-
from obj.task.user import User

'''
管理员：管理员是一种特殊的用户。编写一个名为Admin的类，让它继承你为完成练习3或练习5而编写的User类。
添加一个名为privileges的属性，用于存储一个由字符串（如“can add post”、“can delete post”、“can ban user”等）
组成的列表。编写一个名为show_privileges()的方法,它显示管理员的权限，创建一个Admin实例，并调用这个方法
'''
class Admin(User):

    def __init__(self,first_name,last_name,age,sex,phone,privileges):
        super().__init__(first_name,last_name,age,sex,phone)
        self.privileges = privileges

    def show_privileges(self):
        print("管理员的权限:",end=" ")
        for i in self.privileges:
            print(i, end=",")

if __name__ == '__main__':
    joe = Admin('joe','black',19,'男','18600009999', ['can add post', 'can delete post', 'can ban user'])
    joe.show_privileges()