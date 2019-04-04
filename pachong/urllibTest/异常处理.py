import urllib.request
import urllib.error #异常处理模块

try:
    response=urllib.request.urlopen('http://ibeifeng1.com')
except urllib.error.URLError as e:
    print('发生请求异常--》'+str(e))