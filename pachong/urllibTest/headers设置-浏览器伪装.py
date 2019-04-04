import urllib.request
url='https://www.dianping.com/'
#构建headers
headers={
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36'
}
#构建Request请求对象
req=urllib.request.Request(url=url, headers=headers)
data=urllib.request.urlopen(req).read().decode('UTF-8')
print(data)
# fl=open('大众点评.html','wb')
# fl.write(data)
# fl.close()