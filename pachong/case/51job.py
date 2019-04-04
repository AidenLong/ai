# https://search.51job.com/list/000000,000000,0000,00,9,99,%25E6%2595%25B0%25E6%258D%25AE%25E5%2588%2586%25E6%259E%2590%25E5%25B8%2588,2,1.html
#requests+lxml+pymysql+chardet

import requests
from lxml import etree
import pymysql
import chardet

#1.获取单页html
def get_one_page(url):
    headers={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36'}
    response=requests.get(url,headers=headers)
    response.encoding=chardet.detect(response.content)['encoding']
    return response.text
#2.解析html
def parse_one_page(html):
    #初始化
    result=etree.HTML(html)
    item={}
    item['t1']=result.xpath('//div[@class="el"]/p/span/a/text()') #职位名称
    item['t2']=result.xpath('//div[@class="el"]/span[@class="t2"]/a/text()') #公司名称
    item['t3']=result.xpath('//div[@class="el"]/span[@class="t3"]/text()') #工作地点
    t4=result.xpath('//div[@class="el"]/span[@class="t4"]')
    item['t4']=[]
    for i in t4:
        item['t4'].append(i.xpath('string(.)'))  #职位月薪
    item['t5']=result.xpath('//div[@class="el"]/span[@class="t5"]/text()') #发布时间
    item['href']=result.xpath('//div[@class="el"]/p/span/a/@href')  #详细链接

    #3.数据清洗,处理原始数据
    #(1)去掉职位名称前后空白
    for i in range(len(item['t1'])):
        item['t1'][i]=item['t1'][i].strip()

    #(2)薪资处理
    #定义列表，存储处理后的薪资数据
    zw_low=[] #最低月薪
    zw_height=[] #最高薪资
    #考虑薪资数据可能出现的情况做循环判断
    for xz in item['t4']:
        if xz !="":
            xz=xz.strip().split('-')
            if len(xz)>1:
                if xz[1][-1]=='月' and xz[1][-3]=='万':
                    zw_low.append(float(xz[0])*10000)
                    zw_height.append(float(xz[1][0:-3])*10000)
                elif xz[1][-1]=='年' and xz[1][-3]=='万':
                    zw_low.append(round((float(xz[0])*10000)/12,1))
                    zw_height.append(round((float(xz[1][0:-3])*10000)/12,1))
                elif xz[1][-1]=='月' and xz[1][-3]=='千':
                    zw_low.append(float(xz[0])*1000)
                    zw_height.append(float(xz[1][0:-3])*1000)
                else:
                    zw_low.append(0)
                    zw_height.append(0)
            else:
                if xz[0][-1] =='天' and xz[0][-3]=='元':
                    zw_low.append(xz[0][0:-3])
                    zw_height.append(xz[0][0:-3])
                else:
                    zw_low.append(0)
                    zw_height.append(0)
        else:
            zw_low.append(0)
            zw_height.append(0)
    item['xz_low']=zw_low
    item['xz_height']=zw_height

    #(3) 时间数据处理
    for i in range(len(item['t5'])):
        item['t5'][i]='2018-'+item['t5'][i]
    yield item
#4.存储至mysql
def write_to_mysql(content):
    #建立连接
    conn=pymysql.connect(host='localhost',user='root',passwd='123456',db='test1',charset='utf8')
    cursor=conn.cursor()
    for i in range(len(content['t1'])):
        zwmc=content['t1'][i]
        gsmc=content['t2'][i]
        gzdd=content['t3'][i]
        xz_low=content['xz_low'][i]
        xz_height=content['xz_height'][i]
        ptime=content['t5'][i]
        href=content['href'][i]
        sql='insert into zhaopin values (null,%s,%s,%s,%s,%s,%s,%s)'
        parm=(zwmc,gsmc,gzdd,xz_low,xz_height,ptime,href)
        cursor.execute(sql,parm)
    conn.commit()
    cursor.close()
    conn.close()

#5.函数回调
def main(page):
    url='https://search.51job.com/list/000000,000000,0000,00,9,99,%25E6%2595%25B0%25E6%258D%25AE%25E5%2588%2586%25E6%259E%2590%25E5%25B8%2588,2,'+str(page)+'.html'
    html=get_one_page(url)
    for i in parse_one_page(html):
        print(i)
        # write_to_mysql(i)

#6.回调主函数，完成分页
if __name__ == '__main__':
    for i in range(1,20):
        main(i)