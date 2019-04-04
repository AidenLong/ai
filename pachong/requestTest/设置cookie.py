import requests
f=open('cookie.txt','r')
#初始化cookie字典变量
cookies={}
#按照字符；进行切割读取，返回列表数据，然后遍历
for line in f.read().split(';'):
    #split将参数设置为1，把字符串切割成两部分
    name,value=line.strip().split('=',1)
    #为字典cookie添加内容
    cookies[name]=value
url='http://www.baidu.com/'
res=requests.get(url,cookies=cookies)
data=res.content
f1=open('baidu.html','wb')
f1.write(data)
f1.close()
f.close()