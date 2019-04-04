# 主页url : http://bbs.chinaunix.net/
# 登陆页面url:http://bbs.chinaunix.net/member.php?mod=logging&action=login&logsubmit=yes
# 登陆跳转url:http://bbs.chinaunix.net/member.php?mod=logging&action=login&loginsubmit=yes&loginhash=LF7b3

#无cookie处理登陆
import urllib.request
import urllib.parse
url='http://bbs.chinaunix.net/member.php?mod=logging&action=login&loginsubmit=yes&loginhash=LF7b3'
#使用urlencode编码处理后，再用encode转成utf-8字节流形式数据
postdata=urllib.parse.urlencode({
    'username':'leon_bb',
    'password':'aA123456'
}).encode('utf-8')  #此处登陆可用自己在网站上注册的用户名和密码
data=urllib.request.urlopen(url,postdata).read()
fl=open('chinaUnixlogin.html','wb')
fl.write(data)
fl.close()

url2='http://bbs.chinaunix.net/'
data2=urllib.request.urlopen(url2).read()
fl2=open('chinaUnixBBS.html','wb')
fl2.write(data2)
fl2.close()

# --》出现的问题:登陆后，打开本地第二个网页还是未登录状态
# 问题由来：因为没有设置Cookie, HTTP协议是一个无状态的协议，访问新的网页，会话信息会消失