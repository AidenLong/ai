import urllib.request
import urllib.parse
#http://www.iqianyue.com/mypost/
url='http://www.iqianyue.com/mypost/'
#urlencode() ：将数据使用该函数进行编码处理,
#encode()：将字符串转换成相应编码格式的字节流数据
postdata=urllib.parse.urlencode({
    'name':'37002@qq.com',
    'pass':'a123456'
}).encode('utf-8')
req=urllib.request.urlopen(url, data=postdata).read()
print(req)
# fl=open('post.html','wb')
# fl.write(req)
# fl.close()