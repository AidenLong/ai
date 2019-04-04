import requests
#get请求
r=requests.get('http://www.baidu.com/')
# print(r.content) #字节流形式数据
# print(r.text)  #文本数据

#编码设置
#（1）人工转码
# print(r.content.decode('utf-8'))

#（2）自动处理乱码
import chardet  #字符串/文件编码检测模块
#自动获取到网页的编码,返回类型为字典
print(chardet.detect(r.content))
#encoding:编码方式属性
#自动获取到网页的编码然后赋值给相应内容的encoding属性
r.encoding=chardet.detect(r.content)['encoding']
# print(r.text)
#状态码
print(r.status_code)  #200 正常值