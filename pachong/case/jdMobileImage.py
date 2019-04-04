#1.抓取网页html
#2,根据正则表达式爬去关键内容
#3.根据关键内容，再次使用正则匹配出图片地址
#4.存储图片

import urllib.request
import re
import urllib.error
def craw(url,page):
    html=urllib.request.urlopen(url).read()
    html=str(html)
    #先把所有图片部分的数据取出
    pat1='<div id="plist".+?<div class="clr">'
    result1=re.findall(pat1,html)
    if result1:
        result1=result1[0]
        #常加载正则
        pat2='<img width="220" height="220" data-img="1" src="//(.+?\.jpg)"'
        # 懒加载正则
        pat3='<img width="220" height="220" data-img="1" data-lazy-img="//(.+?.jpg)"'
        imagelist1=re.findall(pat2,result1)
        imagelist2=re.findall(pat3,result1)
        #将所有图片合并
        imagelist=imagelist1+imagelist2
        x=1
        for imageurl in imagelist:
            #对所存的图片进行命名
            imagename='./jd/'+str(page)+str(x)+".jpg"
            #图片地址
            imageurl="http://"+imageurl
            try:
                #获取图片并保存
                urllib.request.urlretrieve(imageurl,filename=imagename)
            except urllib.error.HTTPError as e:
                #hasattr函数判断是否有这些属性
                if hasattr(e,"code"):
                    x += 1
                if hasattr(e,'reason'):
                    x += 1
            x+=1
        print('抓取成功')
    else:
        print('抓取失败，未获得内容')

#分页
for i in range(1,2):
    #url重构
    url='https://list.jd.com/list.html?cat=9987,653,655&page='+str(i)
    craw(url,i)






