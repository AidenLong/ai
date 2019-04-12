# -*- coding:utf-8 -*-

bans = ['刀', '抢', '烟花爆竹']

data = input("顾客所带物品名称：")
if data in bans:
    print('不允许上地铁')
else:
    print('允许上地铁')

for i in range(1, 6):
    print("*" * i)

for i in range(1, 10):
    if i % 2 == 0:
        continue
    print(i, end=' ')

students = ['jack', 'tom', 'john', 'amy', 'kim', 'sunny']
findFlag = False
for i in students:
    if 'amy' == i:
        findFlag = True
        break

if findFlag:
    print('find')
else:
    print('not find')

i = 1
while i < 10:
    if i > 5:
        print("*" * abs(10 - i))
    else:
        print("*" * i)
    i += 1
