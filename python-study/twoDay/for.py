#_*_ conding:utf-8 _*_
'''
    for 变量 in 序列：
        执行的代码块
'''
# list01 = ['joe','susan','jack','Tom']
#遍历列表
# for i in list01: #遍历list01这个列表，将列表中的元素依次复制给 变量 i
#     print(i)  #输出i  直到将所有的元素遍历完毕停止遍历


#用while循环遍历
# i = 0
# while i<len(list01): #获取列表长度
#     print(list01[i]) #通过列表里元素的下表来获取元素
#     i+=1

'''
    练习
'''
# favorite_places = {'张三':['上海','北京','深圳'],
#                    '李四':['张家界','九寨沟','鼓浪屿']}
#
# name = input('请输入姓名:')
# for i in favorite_places:
#     # print(i)   #遍历字典，可以得到字典的 key
#     if name == i: #遍历字典获取key 然后通过key和输入的值判断
#         print(favorite_places[name])
#


# list01 = ['joe','susan','jack','Tom']
# for i in list01:
#     if i == 'susan':
#         #break #终止循环
#         continue
#     print(i)


list01 = ['joe','susan','jack','Tom']
for i in list01:
    if i == 'susan':
        #break #终止循环
        # continue
        i = 'susan2'
        pass #表示什么操作都没有
        print('我上面是pass')
    print(i)



