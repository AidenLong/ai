# -*- coding:utf-8 -*-

#-*- conding:utf-8 -*-
''''''
'''
    open ('文件名','打开的模式')
    r:只读的方式打开，如果文件不存在会提示错误
    w:只写的方式打开，如果文件存在则覆盖，不存在则创建
    a:打开一个文件进行追加内容，如果存在则打开，不存在则创建新的文件
    
    r+:读写，会将文件指针调到文件的头部
    w+:读写，文件不存在直接创建，存在覆盖源文件
    a+:追加读写，会将文件的指针调到文件的末尾
'''
#打开文件
# files = open('python.txt','r') #以只读的模式打开
# print(files)
# #读写操作
#
# #关闭文件
# files.close()

'''
    文件的读写
    read(num):读取文件内容，num表示指定的长度，如果没有则读取所有的数据
'''
#打开文件
# files = open('python.txt','r',encoding= 'utf-8' )
# # print(files)
# #读取文件内容
# # content = files.read()  #读取数据保存在content变量当中
# # content = files.read(10)  #穿入参数会读取指定的长度
# #输出读取的内容
# print(content)
# #关闭文件
# files.close()

'''
    # with open as filename:
    关键字with不在需要访问文件后将其关闭，python会自动判断什么时候去关闭
'''
# with open('python.txt','r',encoding= 'utf-8' ) as files:
#     contnet = files.read()
#     print(contnet)

'''
    readlines
    可以安装行的方式吧整个文件中的内容进行一次性读取，
    并且返回一个列表，其中每一行的数据为一个元素
'''
#打开文件
# files = open('python.txt','r',encoding= 'utf-8' )
# # print(files)
# #读取文件内容
# content = files.readlines()
#
# #输出读取的内容
# print(content)
# #关闭文件
# files.close()

# with open('python.txt','r',encoding= 'utf-8' ) as files:
#     contnet = files.readlines() #按行读取文件内容，保存在列表当中
#     print(contnet)

'''
    逐行读取
'''
#用open结合for循环逐行读取
# files = open('python.txt','r',encoding= 'utf-8')
# i = 1
# for line in files:
#     #没有使用read
#     print('这是第%d行内容:%s'%(i,line))
#     i+=1
# files.close()

#用with结合for
# with open('python.txt','r',encoding= 'utf-8') as files:
#     i = 1
#     for line in files:
#         #没有使用read
#         print('这是第%d行内容:%s'%(i,line))
#         i+=1


#用open结合for,readlines循环逐行读取
# files = open('python.txt','r',encoding= 'utf-8')
# contents = files.readlines()#逐行读取内容
# files.close()#关闭文件
# i = 1
# for line in contents:
#     #没有使用read
#     print('这是第%d行内容:%s'%(i,line))
#     i+=1
# files.close()


# with open('python.txt','r',encoding= 'utf-8') as files:
#     contents = files.readlines()
# i = 1
# for line in contents:
#     #没有使用read
#     print('这是第%d行内容:%s'%(i,line))
#     i+=1


'''
    写入文件
    write
'''

# 以写的方式打开一个文件
# files = open('python.txt','w',encoding='utf-8')
# files = open('python.txt','a',encoding='utf-8')
# content = 'hello,爱你哟'
# content = '元旦一起喝咖啡，看电影可以吗？'
# files.write(content) #写入数据
# files.close()

# with open('python.txt','a',encoding='utf-8') as files:
#     content = '想和你一起去看海'
#     files.write(content)

'''
    tell查看文件指针
'''
# files = open('python.txt','r',encoding='utf-8')
# str = files.read(5)
# print('当前读取的数据是：'+str)
#
# #查看文件的指针
# position = files.tell()
# print('当前的位置是：',position)
#
# str = files.read()
# print('当前读取的数据是：'+str)
#
# #查看文件的指针
# position = files.tell()
# print('当前的位置是：',position)

# files.close()

'''
    seek 设置指针
'''
# files = open('python.txt','r',encoding='utf-8')
# str = files.read(5)
# print('当前读取的数据是：'+str)
#
# # 查看文件的指针
# position = files.tell()
# print('当前的位置是：',position)
#
# # 重新设置文件的指针
# files.seek(2,0)
#
# str = files.read(2)
# print('当前读取的数据是：'+str)
# #查看文件的指针
# position = files.tell()
# print('当前的位置是：',position)
#
# files.close()

# files = open('python.txt','a+')
# files.write('水电费')
# files.seek(0)
# content = files.read()
# print(content)
# files.close()

