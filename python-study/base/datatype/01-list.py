#_*_ conding:utf-8 _*_
'''
    访问列表

'''
# list01 = ['jack','jane','joe','black']
# print(list01[2]) #通过下标

# list01 = ['jack','jane',['leonaldo','joe'],'black']
# l = list01[2]
# print(list01[2][0])
# print(list01[2][0])

# list01 = ['jack','jane',['leonaldo','joe'],'black']
# list01[0] = 'lili'  #通过下标获取到元素，并且给其赋新的值
# print(list01)
#
# list01[2][0] = 'susan'
# print(list01)
#列表是一个可变的类型数据  允许我们对立面的元素进行修改



'''
    列表的操作
        -》append往列表末尾增加元素
        -》insert往列表中指定位置添加元素 （位置，元素）
'''
#append
# list02 = ['jack','jane','joe','black']
# list02.append('susan')
# print(list02)

#insert
# list02 = ['jack','jane','joe','black']
# print('追加前')
# print(list02)
# print('_'*20)
# list02.insert(1,'susan')
# print(list02)


'''
    删除元素
        -》pop 默认删除最后一个
        -》del 通过指定位置删除
        -》remove 通过值删除元素
'''
#pop
# list03 = ['jack','jane','joe','black']
# print('删除前')
# print(list03)
# print('_'*20)
# print(list03.pop())  #执行删除操作 并且返回删除的元素
# print(list03)
# print('继续删除')
# print('_'*20)
# print(list03.pop(1))  #执行删除操作 并且返回删除的元素
# print(list03)


#del
# list03 = ['jack','jane','joe','black']
# print('删除前')
# print(list03)
# print('_'*20)
# del list03     #从内存中将其删除
# print(list03)


#remove
# list03 = ['jack','jane','joe','black']
# print('删除前')
# print(list03)
# print('_'*20)
# list03.remove('jane')  #通过元素的值进行删除
# print(list03)


#查找元素
# list04 = ['jack','jane','joe','black']
# name = 'jack'
# print(name in list04)
#
# name = 'jacks'
# print(name not in list04)

'''
    列表函数
'''
list05 = ['jack','jane','joe','black','joe']

#查看列表的长度  返回列表元素的个数
print(len(list05))

#返回指定元素在列表中出现的次数
print(list05.count('joe'))

#extend
# ll = ['aaa','bbb']
# list05.extend(ll)
# print(list05)
