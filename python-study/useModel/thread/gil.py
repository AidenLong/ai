# -*- coding:utf-8 -*-

import threading

# 加锁
balance = 0


def change(n):
    global balance
    balance += n
    balance -= n


# def run_thread(n):
#     for i in range(1000000):
#         change(n)


lock = threading.Lock()  # 获取线程锁


def run_thread(n):
    for i in range(1000000):
        # 获取锁
        lock.acquire()
        try:
            change(n)
        finally:
            # 释放锁
            lock.release()


t1 = threading.Thread(target=run_thread, args=(4,))
t2 = threading.Thread(target=run_thread, args=(8,))

t1.start()
t2.start()
t1.join()
t2.join()
print(balance)
