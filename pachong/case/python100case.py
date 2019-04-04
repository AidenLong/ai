# -*- coding:utf-8 -*-
import urllib.request
from bs4 import BeautifulSoup

'''
    目标URL：
        http://www.runoob.com/python/python-100-examples.html
    抓取标题、题目、程序分析
'''
# 1.拿到页面连接
def get_herf_list(url):
    # (1)获取网页html
    html = urllib.request.urlopen(url).read()
    # (2)创建bs对象
    soup = BeautifulSoup(html, 'lxml')
    # (3)遍历li得到里面详细页面数据
    urls = soup.find(id='content').find_all('ul')
    for ul in urls:
        lis = ul.find_all('li')
        for li in lis:
            ass = li.find_all('a')
            for a in ass:
                # url 重构
                yield 'http://www.runoob.com' + a['href']


# 2。获取详情页面上数据（标题，题目，程序分析）
def get_indo_text(url_list):
    for url in url_list:
        html = urllib.request.urlopen(url)
        soup = BeautifulSoup(html, 'lxml')
        content = soup.find(id='content')
        if content:
            title = content.find('h1').string
            # 找前三个p标签
            p_list = content.find_all('p', limit=3)
            p_content = ''
            for p in p_list:
                p_content += p.get_text()
            yield (title, p_content)
        else:
            print(url)


# 3.函数回调
url = 'http://www.runoob.com/python/python-100-examples.html'
urls = get_herf_list(url)

content_list = get_indo_text(urls)
# for content in content_list:
#     print(content)

# 4.存储
with open('python100case.text', 'w', encoding='utf-8') as file:
    for title, content in content_list:
        file.write(title + '\n')
        file.write(content + '\n')
        file.write('*' * 20 + '\n')

print('结束')
