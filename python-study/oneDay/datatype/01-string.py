#_*_ conding:utf-8 _*_
# name = 'joe'
# age = 18
# address = '上海'
# print('大家好我叫%s,我今年%d岁，我来自%s'%(name,age,address))
# print('大家好我叫%s'%name)
a = 'bbb'
print('aa{1}aa{0}'.format(a,'nihao'))



'''
    字符串常用函数

'''
# find
# str = 'i love python'
#print(str.find('o'))  #返回的是该字符的位置
# print(str.find('w'))   #返回 -1



#index
# str = 'i love python'
# print(str.index('o'))  #返回的是该字符的位置
# print(str.index('w'))   #返回 报错信息

#count
# str = 'i love python o'
# print(str.count('o'))  #返回的是该字符的位置
# print(str.count('o',2,6))  #指定位置查找
# print(str.count('w'))


#replace
# str = 'i love python o'
# print(str.replace('p','P'))


#split
# str = 'i love python o'
# print(str.split(' '))  #返回一个列表


#upper
# str = 'i love python o'
# print(str.upper())

#title
# str = 'i love python o'
# print(str.title())

#
# str = 'i love python o'
# print(str.capitalize())

#center
str = 'heeloee'
print(str.center(15))

print(str.count('ee'))
str1 = input()
print(str1.upper())