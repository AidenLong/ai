# -*- coding:utf-8 -*-

def calculation(sign, n1, n2):
    if sign == '+':
        print(n1, sign, n2, '=', n1 + n2)
    elif sign == '-':
        print(n1, sign, n2, '=', n1 - n2)
    elif sign == '*':
        print(n1, sign, n2, '=', n1 * n2)
    elif sign == '/':
        print(n1, sign, n2, '=', n1 / n2)
    else:
        print('未知的运算符')


calculation('+', 1, 2)
calculation('-', 1, 2)
calculation('*', 1, 2)
calculation('/', 1, 2)
calculation('~', 1, 2)


def add():
    global students
    name = input('请输入学生名称')
    age = int(input('请输入学生年龄'))
    students[name] = age
    print('添加成功')


def delete():
    global students
    name = input('请输入学生名称')
    del students[name]
    print('删除成功')


def update():
    global students
    name = input('请输入学生名称')
    age = int(input('请输入学生年龄'))
    students[name] = age
    print('修改成功')


def get():
    isAll = input('是否查询全部（是Y，否N）:')
    if isAll == 'Y':
        for name, age in students.items():
            print('查询到%s的年龄是%d' % (name, age))
    else:
        try:
            name = input('请输入学生名称')
            age = students[name]
            print('查询到%s的年龄是%d' % (name, age))
        except KeyError:
            print('没有找到这个同学')


students = {}
while True:
    orders = [1, 2, 3, 4]
    print('1 增加，2 删除，3 修改，4 查询，N 退出')
    order = input('===请选择以上一个选项===')
    if order == 'N':
        break
    else:
        try:
            order = int(order)
            if order not in orders:
                print('输入错误!')
            else:
                index = orders.index(order)
                if index == 0:
                    add()
                if index == 1:
                    delete()
                if index == 2:
                    update()
                if index == 3:
                    get()
        except KeyError:
            print('输入错误!')


def updateList(list1, i, value):
    list1[i] = value

list1 = [1, 2, 3, 4]
print(list1)
updateList(list1, 1, 5)
print(list1)
