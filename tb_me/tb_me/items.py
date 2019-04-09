# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class TbMeItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    pass

class TaobaoGoodItem(scrapy.Item):
    title=scrapy.Field()#标题
    commout=scrapy.Field()#副标题
    price=scrapy.Field() #价格
    confirmGoodsCount=scrapy.Field()#交易成功量
    sendCity=scrapy.Field()#发货城市
    info=scrapy.Field()#商品描述
    itemId=scrapy.Field()#商品编号
    detailCount=scrapy.Field()#评论数
    href=scrapy.Field() #详细页面链接
    tb_price=scrapy.Field()#淘宝价格

    image_urls=scrapy.Field() #图片的URL
    images=scrapy.Field()    #图片磁盘地址
