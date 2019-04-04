import urllib.request
#定义关键词
keyword=input('请输入你想搜索的关键词：')
# 处理中文，进行编码
keyword=urllib.request.quote(keyword)
#url重构
url='http://www.baidu.com/s?word='+keyword
data=urllib.request.urlopen(url).read()
print(data)
# fl=open('baidu2.html','wb')
# fl.write(data)
# fl.close()