#解决方案：
# （1） 导入cookie处理模块 http.cookiejar
# （2） 使用http.cookiejar.CookieJar()创建cookiejar对象
# （3） 使用HTTPCookieProcessor创建cookie处理器，并以此为参数构建opener对象
# （4） 加载为全局默认的opener

import urllib.request
import urllib.parse
import http.cookiejar
url='http://bbs.chinaunix.net/member.php?mod=logging&action=login&loginsubmit=yes&loginhash=LF7b3'
#使用urlencode编码处理后，再用encode转成utf-8字节流形式数据
postdata=urllib.parse.urlencode({
    'username':'leon_bb',
    'password':'aA123456'
}).encode('utf-8')  #此处登陆可用自己在网站上注册的用户名和密码
req=urllib.request.Request(url, postdata)
#使用http.cookiejar.CookieJar()创建cookiejar对象
cjar=http.cookiejar.CookieJar()

#使用HTTPCookieProcessor创建cookie处理器，并以此为参数构建opener对象
opener=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cjar))
#加载为全局默认的opener
urllib.request.install_opener(opener)
data=urllib.request.urlopen(req).read()
file=open('chinaUnixLogin_2.html','wb')
file.write(data)
file.close()

url2='http://bbs.chinaunix.net/'
data2=urllib.request.urlopen(url2).read()
fl2=open('chinaUnixBBS_2.html','wb')
fl2.write(data2)
fl2.close()