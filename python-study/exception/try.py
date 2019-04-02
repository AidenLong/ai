#-*- conding:utf-8 -*-
''''''
'''
    异常
    try:
    except:
    else:
    finally:
'''
# try:
#     print(aaa)  #如果这句话有错，就会捕获到异常
# except ValueError:   #NameError 异常
#     print('变量为定义') #对NameError异常的处理
# except NameError:
#     print('变量为定义')

#捕获异常的具体信息
# try:
#     print(aaa)  #如果这句话有错，就会捕获到异常
# except NameError as e:
#     print(e)#打印具体的异常信息
    # print('变量为定义')

#包含多个异常
# try:
#     # print(aaa)
#     files = open('aaa.txt','r',encoding= 'utf-8')  #如果这句话有错，就会捕获到异常
# except (NameError,FileNotFoundError) as e:
#     print(e)#打印具体的异常信息

#所有异常
# try:
#     print(aaa)
#     files = open('aaa.txt','r',encoding= 'utf-8')  #如果这句话有错，就会捕获到异常
# except Exception as e:
#     print(e)

'''
    esle:没有异常时要执行的语句
'''
# try:
#     # print(aaa)
#     files = open('aaa.txt','r',encoding= 'utf-8')  #如果这句话有错，就会捕获到异常
# except Exception as e:  #有异常时执行
#     print(e)
# else: #没有异常时执行
#     print('美神嘛问题')


'''
    finally 不管有没有异常都会执行的代码块
'''
# try:
#     print('打开文件！')
#     files = open('aaa.txt','w',encoding='utf-8')
#     try:
#         files.write('测试一下行不行')
#     except:
#         print('写入失败')
#     else:
#         print('写入成功')
#     finally:  #不管有没有异常都要执行的代码块
#         print('关闭文件')
#         files.close()
# except Exception as e:
#     print(e)


'''
    练习
    练习
     加法运算：提示用户提供数值输入时，常出现的一个问题是，用户提供的是文本而不是数字。
     在这种情况下，当你尝试将输入转换为整数时，将引发TypeError 异常。编写一个程序，
     提示用户输入两个数字，再将它们相加并打印结果。
     在用户输入的任何一个值不是数字时都捕获TypeError 异常，并打印一条友好的错误消息。
     对你编写的程序进行测试：先输入两个数字，再输入一些文本而不是数字。
'''
# try:
#     num1 = int(input('请输入第一个数字:'))
#     num2 = int(input('请输入第二个数字：'))
# except ValueError:
#     print('请输入一个整数')
# else:
#     print(num1+num2)

