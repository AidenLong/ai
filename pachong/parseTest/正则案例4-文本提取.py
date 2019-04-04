import re
f=open('1.txt','r',encoding='utf-8')
text=f.read()
f.close()
pattern=re.compile('\d+\.\d+')
# pattern='\d+\.\d+'
result=re.findall(pattern,text)
print(result)