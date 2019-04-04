# -*- coding:utf-8 -*-
import urllib.request

#第一种方法
#urllib.request.urlopen(url,data,timeout): 发起请求
#url:需要打开的网址
#data:访问url时候发送的数据包，默认为null
#timeout:等待时长（超时）
response = urllib.request.urlopen('http://www.treejs.cn/v3/demo.php#_101')
# decode()：解码  把字节流形式数据以相应的格式转换为字符串
print(response.read().decode('UTF-8'))

#第二种方法 urlretrieve(url,filename)
urllib.request.urlretrieve('http://www.treejs.cn/v3/demo.php#_101', 'ztree.html')
#清除缓存
urllib.request.urlcleanup()
