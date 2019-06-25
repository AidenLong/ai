# -*- coding:utf-8 -*-

file_name = 'peoples.txt'


def add(name, phone, qq, address, work_address):
    """
    添加联系人信息
    :param name:
    :param phone:
    :param qq:
    :param address:
    :param work_address:
    :return:
    """
    with open(file_name, 'a+', encoding='utf-8') as file:
        file.write(name + ',' + phone + ',' + qq + ',' + address + ',' + work_address + '\n')
        file.flush()
    print('添加 %s 信息成功' % name)


def delete(name=None, phone=None):
    """
    根据名称或手机号删除信息
    :param name:
    :param phone:
    :return:
    """
    del_flag = False
    peoples = get_peoples()
    if not name and not phone:
        print('联系人姓名和电话为空，删除失败')
        return
    else:
        for people in peoples:
            if name and name == people.split(',')[0]:
                peoples.remove(people)
                del_flag = True
                break
            if phone and phone == people.split(',')[1]:
                peoples.remove(people)
                del_flag = True
                break

    update_peoples(peoples)
    if del_flag:
        print('删除 %s 信息成功' % (name if name else phone))
    else:
        print('删除失败，未找到用户信息。')


def update(name, phone, qq, address, work_address):
    """
    根据名称更新信息
    :param name:
    :param phone:
    :param qq:
    :param address:
    :param work_address:
    :return:
    """
    update_flag = False
    peoples = get_peoples()
    for people in peoples:
        if name == people.split(',')[0]:
            peoples.remove(people)
            peoples.append(name + ',' + phone + ',' + qq + ',' + address + ',' + work_address + '\n')
            update_flag = True
            break
    update_peoples(peoples)
    if update_flag:
        print('更新 %s 信息成功' % (name if name else phone))
    else:
        print('更新失败，未找到用户信息。')


def get(name=None, phone=None, qq=None, address=None, all=None):
    """
    查询
    :param name:
    :param phone:
    :param qq:
    :param address:
    :param all:
    :return:
    """
    peoples = get_peoples()
    if not name and not phone and not qq and not address and all:
        return peoples
    results = []
    for people in peoples:
        if name and name in people.split(',')[0]:
            results.append(people)
        if phone and phone in people.split(',')[1]:
            results.append(people)
        if qq and qq in people.split(',')[2]:
            results.append(people)
        if address and address in people.split(',')[3]:
            results.append(people)
    if len(results) == 0:
        print('未查询到接对应的数据！')
    return results


def input_info():
    """
    输入用户信息
    :return:
    """
    name = input('请输入姓名：')
    while not name:
        name = input('姓名不能为空，请重新输入姓名：')
    phone = input('请输入电话号码：')
    while not phone:
        phone = input('电话号码不能为空，请重新请输入电话号码：')
    qq = input('请输入QQ：')
    address = input('请输入家庭地址：')
    work_address = input('请输入工作单位：')
    return name, phone, qq, address, work_address


def get_peoples():
    """
    从文件中获取联系人信息
    :return:
    """
    with open(file_name, 'r', encoding='utf-8') as file:
        return file.readlines()


def update_peoples(peoples):
    """
    更新保存联系人信息文件
    :param peoples:
    :return:
    """
    with open(file_name, 'w', encoding='utf-8') as file:
        file.writelines(peoples)
        file.flush()


while True:
    option = input('选择操作类型：1（添加）、2（删除）、3（修改）、4（查询）、其他键（退出）')
    if option == '1':
        print('=' * 5, '添加用户', '=' * 5)
        name, phone, qq, address, work_address = input_info()
        add(name, phone, qq, address, work_address)
    elif option == '2':
        print('=' * 5, '删除用户', '=' * 5)
        option = input('选择删除用户方式，1（根据姓名）、2（根据联系电话）')
        if option == '1':
            name = input('请输入姓名：')
            delete(name=name)
        else:
            phone = input('请输入电话号码：')
            delete(phone=phone)

    elif option == '3':
        print('=' * 5, '修改用户', '=' * 5)
        name, phone, qq, address, work_address = input_info()
        update(name, phone, qq, address, work_address)

    elif option == '4':
        print('=' * 5, '查询用户', '=' * 5)
        option = input('选择删除用户方式，1（查询全部）、2（根据姓名）、3（根据联系电话）、4（根据qq）、5（根据家庭地址）')
        if option == '1':
            peoples = get(all=True)
        elif option == '2':
            name = input('请输入姓名：')
            peoples = get(name=name)
        elif option == '3':
            phone = input('请输入电话号码：')
            peoples = get(phone=phone)
        elif option == '4':
            qq = input('请输入qq：')
            peoples = get(qq=qq)
        elif option == '4':
            address = input('请输入家庭地址：')
            peoples = get(address=address)
        else:
            peoples = get(all=True)
        print('查询出的结果')
        for people in peoples:
            attr = people.split(',')
            print('姓名%s，电话%s，qq%s，家庭地址%s，工作地址%s' % (attr[0], attr[1], attr[2], attr[3], attr[4]))
    else:
        print('=' * 5, '退出系统', '=' * 5)
        break
