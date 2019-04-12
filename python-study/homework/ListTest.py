# -*- coding:utf-8 -*-

students = ['小明', '小红', '小刚']

for student in students:
    print(student)

for student in students:
    print('%s，你好' % student)

students.append('王老师')
students.insert(0, '班主任')

for student in students:
    print(student)

student = students[2]
students.remove(student)
print('非常抱歉，%s，您该走了' % student)
