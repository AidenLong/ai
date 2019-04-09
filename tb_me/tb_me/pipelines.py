# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html
import pymysql
from scrapy.exceptions import DropItem
from tb_me import settings


# 爬虫管道设计中
# （1）数据清洗###
# （2）数据去重
# 不仅要使用集合方式处理重复数据，链接数据验证数据是否存在
class DataOnePipeline(object):

    # 初始化，设置一个集合用来存储已经抓取过得数据集（使用唯一标识的数据，来判断是否已经抓取过）
    # 配置mysql基础数据
    def __init__(self, mysql_host, mysql_user, mysql_passwd, mysql_db, mysql_charset):
        self.ids_seen = set()
        self.mysql_host = mysql_host
        self.mysql_user = mysql_user
        self.mysql_passwd = mysql_passwd
        self.mysql_db = mysql_db
        self.mysql_charset = mysql_charset

    # 爬虫默认调用的方法，基础的预处理
    @classmethod
    def from_crawler(cls, crawler):
        # 调用实例化
        return cls(
            mysql_host=crawler.settings.get('MYSQL_HOST', 'localhost'),
            mysql_user=crawler.settings.get('MYSQL_USER', 'root'),
            mysql_passwd=crawler.settings.get('MYSQL_PASSWD', '123456'),
            mysql_db=crawler.settings.get('MYSQL_DB', 'taobao'),
            mysql_charset=crawler.settings.get('MYSQL_CHARSET', 'utf8')
        )

    # 爬虫运行时调用
    def open_spider(self, spider):
        self.conn = pymysql.connect(host=self.mysql_host, user=self.mysql_user, passwd=self.mysql_passwd,
                                    db=self.mysql_db, charset=self.mysql_charset)

    # 爬虫关闭时调用
    def close_spider(self, spider):
        self.conn.close()

    # 判断唯一标识的数据是否在集合内
    def process_item(self, item, spider):
        # 判断数据是否存在于集合内
        if item['itemId'] in self.ids_seen:
            raise DropItem("Duplicate item found: %s" % item)
        # 不存在，则在数据库中进行验证
        else:
            # 判断item['itemId']是否出现在数据库中
            cusor = self.conn.cursor()
            # 数据去重判断
            sql = 'select * from goods where itemid=%s'
            cusor.execute(sql, (item['itemId'],))
            result = cusor.fetchone()
            # 如果result他存在，则表示数据库中已经有数据了
            if result:
                # 抛出一个异常
                raise DropItem("Duplicate item found: %s" % item)
            # 数据存储
            else:
                # 假如去除空格后，title为空，咋把cimmout的数据交给title
                if item['title'].strip() == '':
                    item['title'] = item['commout'].strip()
                # 设置自动提交
                self.conn.autocommit(1)
                sql = 'insert into goods (title,commout,price,confirm_goods_count,send_city,info,itemid,detail_count,href,tb_price) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'
                cusor.execute(sql, (
                    item['title'].strip(), item['commout'].strip(), item['price'], item['confirmGoodsCount'],
                    item['sendCity'], item['info'], item['itemId'], item['detailCount'], item['href'],
                    item['tb_price']))
                self.conn.commit()
            self.ids_seen.add(item['itemId'])
            return item


# （3）数据存储
# class MysqlSavePipeline(object):
#     def process_item(self, item, spider):
#         #1.建立数据库链接
#
#         #2.判断数据是否存在,判断itemId是否存在记录
#
#         #3.生成sql语句执行
#
#         return item

import os
import requests


# 使用pip install requests  进行安装
# （4）图片存储
class ImageDownloadPipeline(object):
    # 管道的默认入口
    def process_item(self, item, spider):
        if 'image_urls' in item:
            images = []

            # 设置路径
            dir_path = '%s/%s' % (settings.IMAGES_STORE, item['itemId'])

            # 判断路径路径是否存在
            if not os.path.exists(dir_path):
                # os.mkdier()这个只能生成一级目录，
                # os.makedires可以生成多级目录
                os.makedirs(dir_path)
            # 循环图片路径
            for image_url in item['image_urls']:
                # 为了避免数据异常，将url链接中域名后的链接，将  /  转换为下划线 用作最终存储的文件名
                us = image_url.split('/')[3:]
                file_name = '_'.join(us)
                file_path = '%s/%s' % (dir_path, file_name)
                # 路径存储到images中
                images.append(file_path)
                # 判断是否已经存在文件
                if os.path.exists(file_path):
                    continue
                # 上下文with
                with open(file_path, 'wb') as handle:
                    if 'https:' not in image_url:
                        image_url = 'https:' + image_url
                    # 请求图片路径 ，stream 数据流
                    response = requests.get(image_url, stream=True)
                    # 文件流写入, 使用1024的模式提取数据
                    for block in response.iter_content(1024):
                        # 数据提取完成
                        if not block:
                            break
                        # 写入文件
                        handle.write(block)
            # 图片路径结果返回出来
            item['images'] = images
        return item
