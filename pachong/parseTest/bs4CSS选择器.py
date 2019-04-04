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

#1.根据标签查询标签对象
res1=soup.select('a')
# print(res1)

#2.根据ID属性查询标签对象
res2=soup.select('#link2')
# print(res2)

#3.根据Class属性查询标签对象
res3=soup.select('.sister')
# print(res3)

#4.属性选择
res4=soup.select("a[href='http://example.com/tillie']")
# print(res4)

#5.包含选择
res5=soup.select('p a#link3')
# print(res5)
# 6.得到标签内容
res6=soup.select('p a.sister')
print(res6[0].string)
print(res6[0].get_text())