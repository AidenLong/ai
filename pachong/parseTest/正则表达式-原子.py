import re
# 一.原子

#（1）普通字符
pattern='yue'
string='http://www.iqianyue.com/'
result=re.search(pattern,string)
print(result)

# (2)非打印字符作为原子
pattern2='\n'
string2='''http://www.iqianyue.com/
http://www.baidu.com'''
result2=re.search(pattern2,string2)
# print(result2)

#（3）通用字符为原子
string3='abcdfphp345python_py'
pattern3='\d\dpython\w'
result3=re.search(pattern3,string3)
# print(result3)

#(4)原子表（字符集） []
#定义一组地位平等的原子，然后匹配的时候回取该院子边中任意一个原子进行匹配
string4='abcdfphp345python_py'
pattern4='p[h3y]'
pattern4_2='p[^h3]' #取补集（除了h或3）
# result4=re.findall(pattern4,string4)
result4_2=re.search(pattern4,string4)
# print(result4)
# print(result4_2)
result4_3=re.search(pattern4_2,string4)
# print(result4_3)
