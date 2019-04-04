import re
string = "<a href='http://www.baidu.com'>百度首页</a>"
pattern='[a-zA-Z]+://[^\s]*[.com|.cn]'
result=re.search(pattern,string)
print(result)