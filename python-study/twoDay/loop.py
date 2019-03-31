#_*_ conding:utf-8 _*_
'''
    while 条件:
        条件成立时执行的代码块
    else:
        条件不成立时执行的代码块
'''
# a = 0  #初始化变量
# while a < 5:
#     print('爱你一万年')
#     a+=1  #进行累加   每一次循环自动加1
# else:   #当条件不成立的时候
#     print(a)
#     print('我是至尊宝')


'''
    跳转语句
    break：终止循环，不管循环提哦案件是否为真，不再循环下去
'''
# a = 0  #初始化变量
# while a < 5:  #判断条件  a = 0
#     if a == 3: #判断条件
#         print('紫霞仙子：别说话，吻我！')
#         break  #当a = 3
#         # 进入if,然后输出一句话，然后break跳出整个循环，
#         # 不管条件是否为真
#     print('至尊宝:爱你一万年')
#     a+=1  #进行累加   每一次循环自动加1

'''
    示例
'''
# while True: #给个条件 为True
#     a = input('你是否要退出程序(y/n):')
#     print(a)
#     if a == 'y':#当用户的输入为y的时候进入if 然后break 跳出循环
#         break

'''
    continue
    跳出当前循环，直接开始下一次循环
'''
# a = 0
# while a <10:
#     a += 1  # 累加
#     print('第%d圈开始'%a)
#     print('好累哈')
#     if a == 5:
#         print('蹭老师不注意，后半圈没跑')
#         print()
#         continue
#     print('第%d圈结束' % a)
#     print()

'''
    break:跳出整个循环，不管条件是否为真
    continue：跳出当前循环，直接回到起点开始下一次循环
'''

