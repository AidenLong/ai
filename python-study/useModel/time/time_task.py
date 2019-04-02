# -*- coding:utf-8 -*-
'''
    1.将字符串的时间"2017-10-10 23:40:00"转换为时间戳和时间元组
    2.字符串格式更改。如提time = "2017-10-10 23:40:00",想改为 time= "2017/10/10 23:40:00"
    3.获取当前时间戳转换为指定格式日期
    4.获得三天前的时间
'''
import time

timeStr = "2017-10-10 23:40:00"
# 时间字符串转元组
timeTulp = time.strptime(timeStr, '%Y-%m-%d %H:%M:%S')
print(timeTulp)
# 元组转时间戳
print(time.mktime(timeTulp))

# 元组转固定格式字符串
print(time.strftime('%Y/%m/%d %H:%M:%S', timeTulp))

now = time.time()
# 时间戳转元组
timeTulp = time.localtime(now)
# 元组转固定格式字符串
print(time.strftime('%Y/%m/%d %H:%M:%S', timeTulp))

now -= 60 * 60 * 24 * 3
timeTulp = time.localtime(now)
print(time.strftime('%Y/%m/%d %H:%M:%S', timeTulp))