#-*- conding:utf-8 -*-
''''''
'''
    引入time模块
'''
import time
#回格林威治西部的夏令时地区的偏移秒数
# print(time.altzone)

#接受时间元组并返回一个可读的形式为"Tue Dec 11 18:07:14 2008"的24个字符的字符串。
# print(time.asctime())  #返回可读形式的当前时间
# print(time.asctime((2017,12,12,12,12,12,3,340,1)))

#返回进程时间
# print(time.clock())
# print(time.clock())
# print(time.clock())

#@@@@@返回当前时间的时间戳
# print(time.time())
# times = time.time()

#@@@@@@@获取读形式的当前时间
# print(time.ctime(times))

# print(time.gmtime()) #返回时间元祖  返回的是格林威治时间元祖

#@@@@@@返回时间元祖 返回的是当前时间
# print(time.localtime())

'''
    时间戳转换为时间元祖，将时间元祖转换为时间字符串
'''
#获取当前时间戳
times = time.time()

#将时间戳转换为时间元祖
# print(time.localtime(times))
formatTime = time.localtime()
print(formatTime)
#接收以时间元组，并返回指定格式可读字符串表示的当地时间，格式由fmt决定。
print(time.strftime(u'%Y-%m-%d %H:%M:%S',formatTime))


'''
    time.strptime 将时间字符串转换为时间元祖
'''

# times = '2017-12-12 12:12:12'
#转换为时间元祖
# formatTime = time.strptime(times,'%Y-%m-%d %H:%M:%S')
# print(formatTime)

#将时间元祖转换为时间戳i mktime
# print(time.mktime(formatTime))


'''
    sleep 推迟调用线程的运行，secs指秒数
'''
# for i in range(1,2):
#     print('让子弹飞一会')
#     time.sleep(2)
#     print('子弹在飞')
#     time.sleep(2)
#     print('子弹到了')
