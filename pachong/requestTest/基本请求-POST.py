import requests
#构建上传到网页的数据
data={
    'name':'leon',
    'pass':'aabbcc'
}
r=requests.post('http://www.iqianyue.com/mypost/',data=data)
f=open('login.html','wb')
f.write(r.content)
f.close()