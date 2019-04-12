# -*- coding:utf-8 -*-

def pri(students):
    for student in students:
        print(student)
        print(students[student])
    print('*' * 20)


students = {'小明': {'年龄': 18, '性别': '男', '学号': 1},
            '小刚': {'年龄': 19, '性别': '男', '学号': 2}}
pri(students)
# 添加
students['小红'] = {'年龄': 11, '性别': '女', '学号': 3}
pri(students)
# 修改
students['小红'] = {'年龄': 15, '性别': '女', '学号': 3}
pri(students)
# 删除
del students['小红']
pri(students)

dict1 = {'看书': '读书', '飞机': '重于空气的固定翼航空器,用螺旋桨或高速喷气推进并且受空气对其翼面之动力反作用所支承'}
while True:
    key = input('请输入一个名词（退出程序请输入N）：')
    if key == 'N':
        print('退出程序')
        break
    value = input('请输入上面名词的解释：')
    dict1[key] = value

    print('现在字典中存在的数据如下')
    for k, v in dict1.items():
        print(k, ':', v)
