# -*- coding:utf-8 -*-

'''
    https://maoyan.com/board/4?offset=10
    抓取排名、海报、电影名、主演、上映时间、评分
'''

import re
import requests
import pymysql
from requests.exceptions import RequestException


# 1.请求单页内容拿到html
def get_one_page(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36'}
        req = requests.get(url, headers=headers)
        if req.status_code == 200:
            return req.text
    except RequestException as e:
        print('发生异常', str(e))


# 2.解析html（排名、海报、电影名、主演、上映时间、评分）
def parse_one_page(page):
    # 创建正则
    # 使用re.S可以使 元字符.匹配到换行符
    pattern = re.compile('<dd>.*?board-index.*?>(\d+)</i>.*?data-src="(.*?)".*?name"><a'
                         + '.*?>(.*?)</a>.*?star">(.*?)</p>.*?releasetime">(.*?)</p>'
                         + '.*?integer">(.*?)</i>.*?fraction">(.*?)</i>.*?</dd>', re.S)
    items = re.findall(pattern, page)
    # print(items)

    # 3.数据处理
    for item in items:
        yield {
            'index':item[0],
            'image':item[1],
            'title':item[2],
            'actor':item[3].strip()[3:], # 去掉前后空格，切片
            'time':item[4].strip()[5:],
            'score':item[5] + item[6],
        }

# 4.数据存储
def write_to_mysql(content):
    conn = pymysql.connect(host='localhost')

# 5.函数回调
def main(offset):
    url = 'http://maoyan.com/board/4?offset=' + str(offset)
    html = get_one_page(url)
    # print(html)
    items = parse_one_page(html)
    for item in items:
        print(item)

if __name__ == '__main__':
    for i in range(0, 10):
        main(i * 10)