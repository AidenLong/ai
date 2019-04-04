import urllib.request
#定义关键词
keyword=input('请输入你想搜索的关键词：')
#url重构
url='http://www.baidu.com/s?word='+keyword
#发送请求
req=urllib.request.urlopen(url)
#读取数据
result=req.read()
print(result)
#数据存储
# fl=open('baidu1.html','wb')
# fl.write(result)
# fl.close()