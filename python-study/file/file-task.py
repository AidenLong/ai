#-*- conding:utf-8 -*-
''''''
'''
    在文本编辑器中新建一个文件，写几句话来总结一下你至此学到的Python 知识，
    其中每一行都以“In Python you can”打头。将这个文件命名为learning_python.txt，
    并将其存储到为完成本章练习而编写的程序所在的目录中。编写一个程序，它读取这个文件，
    并将你所写的内容打印三次：第一次打印时读取整个文件；第二次打印时遍历文件对象；
    第三次打印时将各行存储在一个列表中。
'''
# for i in range(3):
#     if i == 0:
#         with open('learning_python.txt','r',encoding='utf-8') as files:
#             content = files.read()
#             print('one')
#             print(content)
#     if i == 1:
#         with open('learning_python.txt','r',encoding='utf-8') as files:
#             print('two')
#             for line in files:
#                 print(line)
#     if i == 2:
#         with open('learning_python.txt','r',encoding='utf-8') as files:
#             content = files.readlines()
#             print('three')
#             print(content)

'''
下面是一个简单的示例，演示了如何将句子中的'dog'替换为'cat'：
>>> message = "I really like dogs."
>>> message.replace('dog', 'cat')
'I really like cats.'
读取你刚创建的文件learning_python.txt 中的每一行，将其中的Python 都替换为另
一门语言的名称，如C。将修改后的各行都打印到屏幕上。块外打印它们。

'''
#读取内容
# file1 = open('learning_python.txt','r',encoding='utf-8')
# content1 = file1.read()
# file1.close()
#
# #写入内容并且读取新的内容
# file2 = open('learning_python.txt','w+',encoding='utf-8')
# file2.write(content1.replace('Python','C'))#写入的时候替换内容
# file2.seek(0)#重置指针导开头
# content2 = file2.read()#读取所有的内容
# print(content2)
# file2.close()


'''
    访客：编写一个程序，提示用户输入其名字；用户作出响应后，将其名字写
入到文件guest.txt 中。

'''
while True:
    name = input('请输入您的姓名:')
    if name == 'n':
        break
    with open('guest.txt','a+',encoding='utf-8') as files:
        files.write(name)
        files.write('\n')
        files.seek(0)
        content = files.read()
        print(content)