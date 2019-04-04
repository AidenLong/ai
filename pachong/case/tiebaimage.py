# -*- coding:utf-8 -*-
# https://tieba.baidu.com/p/4394707943?fr=good

import re
import urllib.request

'''
    爬虫获取图片路径并下载
'''
# 1.发送请求获取html代码
def getHtmlContent(url):
    page = urllib.request.urlopen(url)
    return page.read().decode('utf-8')


# 2.从html中解析出图片的url
def getJPGUrls(html):
    jpgReg = re.compile('img class="BDE_Image".*?src="(.*?.jpg)".*?>')
    jpgs = re.findall(jpgReg, html)
    return jpgs


# 3.用图片url保存成文件名
def downladJPGS(jpgurl, filename):
    urllib.request.urlretrieve(jpgurl, filename)


# 4.批量下载图片，保存到当前目录
def batchDownloadJPGS(jpgs, path='./meitu/'):
    count = 1
    for jpg in jpgs:
        downladJPGS(jpg, ''.join([path, '{0}.jpg']).format(count))
        print('下载完成第%d张图片' % count)
        count += 1


# 5.函数回调
def download(url):
    html = getHtmlContent(url)
    jpgs = getJPGUrls(html)
    batchDownloadJPGS(jpgs)


if __name__ == '__main__':
    url = 'https://tieba.baidu.com/p/4394707943?fr=good'
    download(url)
