# -*- coding:utf-8 -*-
import time
import threading


def music(name, loop):
    for i in range(loop):
        print('listen music %s %s %s' % (name, time.ctime(), threading.Thread.getName(t1)))
        time.sleep(1)


def movie(name, loop):
    for i in range(loop):
        print('look movie %s %s %s' % (name, time.ctime(), threading.Thread.getName(t2)))
        time.sleep(1)


# 单线程
# if __name__ == '__main__':
#     music('music', 2)
#     movie('movie', 3)

# 多线程
t1 = threading.Thread(target=music, args=('music', 2), name='musicThread')
t2 = threading.Thread(target=movie, args=('movie', 2), name='movieThread')

if __name__ == '__main__':
    t1.start()
    t2.start()
    t2.join()
    print('主线程运行结束 %s'% time.ctime())
