import re
string='apysgdpythonsadhhkkkkspppp'
pattern='.py'
result=re.match(pattern,string)
# if result:
#     print(result.group())
# else:
#     print('不匹配')

# re.sub(正则,替换后的字符，处理的字符串，(替换的次数))
string2='hellomypythonhispythonourpythonend'
pattern2='python'
result1=re.sub(pattern2,'php',string2) #全部替换
result2=re.sub(pattern2,'php',string2,2)
# print(result1)
print(result2)