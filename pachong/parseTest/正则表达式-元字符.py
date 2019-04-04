# （1）任意匹配元字符  .
# 可以匹配除了换行符以外任意一个字符
import re
string='abcdphppython12ab3py_'
pattern='.python...'
result=re.search(pattern,string)
# print(result)

# (2)边界限制元字符 ^ $
# ^ : 匹配字符串开始的位置
#$ : 匹配字符串结束的位置

string2_1='abcdphppython12ab3py_'
string2_2='bbcdpthh1234'
pattern2='^abc'
result2_1=re.search(pattern2,string2_1)
# print(result2_1)
result2_2=re.search(pattern2,string2_2)
# print(result2_2)

# (3)次数限定元字符
# * : 可以匹配到前一个字符至少0次 最多无限次 -贪婪
string3='abcdphppython12ab3py_'
pattern3='py.*12'
result3=re.search(pattern3,string3)
# print(result3)

# + : 可以匹配到前一个字符至少1次 最多无限次 -贪婪
string4='abcdphppython12ab3py_'
pattern4='py.+h'
result4=re.search(pattern4,string4)
# print(result4)

# ? : 可以匹配到前一个字符至少0次 最多1次 -非贪婪（匹配尽可能少的次数）
string5='abcdlhlpython12ab3py_'
pattern5_1='p.*y'
pattern5_2='p.*?y'
result5_1=re.search(pattern5_1,string5)
result5_2=re.search(pattern5_2,string5)
# print(result5_1)
# print(result5_2)

# (4)模式选择符  |
# 使用模式选择符，可以设置多个模式，匹配时，可以从中任意一个模式匹配
pattern6='python|lhl'
string6='abcdlhlpython12ab3py_'
result=re.search(pattern6,string6)
# print(result)

# (5) 模式单元符 （）
# 可以使用（）将一些院子组合成一个大原子使用。
string7='abcdlhlpythcd12acdpy_cd'
patter7_1='(cd){1,3}'
result7=re.findall(patter7_1,string7)
print(result7)
