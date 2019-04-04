import re
string = "<a href='http://www.baidu.com'>百度首页</a><br><a href='mailto:c-e+o@iqi-anyue.com.cn'>电子'邮箱'</a>"
#通过标签结构分析获取到href属性值，再拿到电子邮箱
pattern="<br><a href='mailto:(.*?)'>"
# findall方法匹配，如果正则里有圆括号，获取是圆括号内的字符
result=re.findall(pattern,string)
print(result[0])

#通过电子邮箱规则，指定正则
pattern2='[a-zA-Z+-]+@[a-zA-Z-]+\.\w+([.-]\w+)*'
result2=re.search(pattern2,string)
print(result2)