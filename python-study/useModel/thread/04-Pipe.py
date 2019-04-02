#-*- conding:utf-8 -*-
import multiprocessing
import time
#PIPE 管道通信
def put(p):
   for value in ['A', 'B', 'C']:
       print ('发送 %s 到 pipe...' % value)
       p[1].send(value)
       time.sleep(2)

# 读数据进程执行的代码:
def get(p):
   while True:
       value = p[0].recv()
       print ('从 pipe 接受 %s .' % value)

if __name__=='__main__':
   # 父进程创建Queue，并传给各个子进程：
   # p = multiprocessing.Pipe()
   p = multiprocessing.Pipe(duplex=False) #左收右发
   pw = multiprocessing.Process(target=put, args=(p,))
   pr = multiprocessing.Process(target=get, args=(p,))
   # 启动子进程pw，写入:
   pw.start()
   # 启动子进程pr，读取:
   pr.start()
   # 等待pw结束:
   pw.join()
   # pr进程里是死循环，无法等待其结束，只能强行终止:
   pr.terminate()