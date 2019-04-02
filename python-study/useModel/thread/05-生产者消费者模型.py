#-*- conding:utf-8 -*-
import threading
import time
import queue

#生产者消费者模型

q = queue.Queue(maxsize=10)
def producer(name):  # 生产者
    count = 1
    while True:
        q.put("骨头%s" % count)
        print("生产了骨头", count)
        count += 1
        time.sleep(0.5)


def consumer(name):  # 消费者
    while True:
        print("[%s]取到[%s]并且吃了它..." % (name, q.get()))
        time.sleep(1)


p = threading.Thread(target=producer, args=("Tim",))
c1 = threading.Thread(target=consumer, args=("King",))
c2 = threading.Thread(target=consumer, args=("Wang",))

p.start()
c1.start()
c2.start()