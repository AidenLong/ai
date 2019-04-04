from bs4 import BeautifulSoup

html_doc = """
<html><head><title>The Dormouse's story</title></head>
<body>
<p class="title"><b>The Dormouse's story</b></p>

<p class="story">Once upon a time there were three little sisters; and their names were
<a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>,
<a href="http://example.com/lacie" class="sister" id="link2">Lacie</a> and
<a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>;
and they lived at the bottom of a well.</p>

<p class="story">...</p>
"""

soup=BeautifulSoup(html_doc,'lxml')
#1.标签名
# 查询所有a标签
res1=soup.find_all('a')
print(res1)

#2.正则表达式
import re
#查找所有包含d字符的标签
res2=soup.find_all(re.compile('d+'))
# print(res2)

#3.列表：选择
#查找所有的title标签或者a标签
res3=soup.find_all(['title','a'])
# print(res3)

#4.关键字参数
#查询属性id="link1"的标签
res4=soup.find_all(id='link1')
# print(res4)

#5.内容匹配
res5=soup.find_all(text='Elsie') #直接匹配内容中的字符
# print(res5)
#通过正则表达式进行模糊匹配
res6=soup.find_all(text=re.compile('Dormouse+'))
# print(res6)

#6.嵌套选择
# for i in soup.find_all('p'):
#     print(i.find_all('a'))