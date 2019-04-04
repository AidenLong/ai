import re
string = "021-6728263653682382265236"
pattern='\d{4}-\d{7}|\d{3}-\d{8}'
result=re.findall(pattern,string) #返回列表
print(result[0])