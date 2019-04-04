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
#创建一个BS对象
soup=BeautifulSoup(html_doc,'lxml')
#打印类型
# print(type(soup))
#结构化输出
# print(soup.prettify())

#1.获取标签（只能获取到第一条的标签）
# print(soup.title)
# print(soup.p)
# print(soup.a)

#2.获取属性
# print(soup.a.attrs) #返回字典
# print(soup.a['id']) #得到指定属性的值

#3.获取标签的内容
# print(soup.head.string) #如果标签中只有一个子标签，返回子标签中文本内容
# print(soup.body.string) #如果标签中有多个字标签，返回None

#4.操作子节点
# print(soup.p.contents) #得到p标签的所有子节点

# print(soup.p.children) #得到匹配的第一个p标签的子节点列表迭代器
for i in soup.p.children:
    print(i)






