#-*- conding:utf-8 -*-
import multiprocessing
import time
#queue 跨进程通信
def put(q):
   for value in ['A', 'B', 'C']:
       print ('发送 %s 到 queue...' % value)
       q.put(value)  #通过put发送
       time.sleep(2)

## 读数据进程执行的代码:
def get(q):
   while True:
       value = q.get(True) #接受队列中的数据
       print ('从 queue 接受 %s .' % value)

if __name__=='__main__':
   # 父进程创建Queue，并传给各个子进程：
   q = multiprocessing.Queue()
   pw = multiprocessing.Process(target=put, args=(q,))
   pr = multiprocessing.Process(target=get, args=(q,))
   # 启动子进程pw，写入:
   pw.start()
   # 启动子进程pr，读取:
   pr.start()
   # 等待pw结束:
   pw.join()
   # pr进程里是死循环，无法等待其结束，只能强行终止:
   pr.terminate()