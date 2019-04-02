# -*- conding:utf-8 -*-
import os
import multiprocessing
import time


def work(n):
    print('run work (%s) ,work id %s' % (n, os.getpid()))
    time.sleep(5)
    print('work (%s) stop ,work id %s' % (n, os.getpid()))


if __name__ == '__main__':
    print('Parent process %s.' % os.getpid())
    # 创建进程池
    p = multiprocessing.Pool(3)
    for i in range(5):
        # 创建5个进程，一次进入进程池
        p.apply_async(work, args=(i,))
    p.close()
    p.join()

# def music(name,loop):
#     print(time.ctime())
#     for i in range(loop):
#         time.sleep(2)
#         print('您现在正在听的音乐是%s'%name)
#
# def movie(name,loop):
#     print(time.ctime())
#     for i in range(loop):
#         time.sleep(2)
#         print('您现在正在看的电影是%s'%name)
#
# if __name__=='__main__':
#     pool=multiprocessing.Pool(2)
#     pool.apply_async(func=music,args=('花太香',3))
#     pool.apply_async(func=movie,args=('王牌特工',4))
#     pool.apply_async(func=music, args=('爱的故事上集', 2))
#     pool.close()
#     # pool.terminate()
#     # 比较危险,不要轻易用,直接杀死进程池
#     #join阻塞主进程,当子进程执行完毕的时候会继续往后执行,使用join必须在进程池使用terminate或者close
#     pool.join()
#     print('结束时间是%s'%time.ctime())
