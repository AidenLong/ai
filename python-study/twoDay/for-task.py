#_*_ conding:utf-8 _*_

'''
    if 练习01
    利用条件运算符的嵌套来完成此题：学习成绩>=90分的同学用A表示，60-89分之间的用B表示，60分以下的用C表示。
'''
# score = int(input('请输入成绩：'))  #转换为int类型
# if score >= 90:
#     print('你的成绩为A')
# elif 60 <= score <=89:
#     print('你的成绩是B')
# else:
#     print('你的成绩是C')



'''
    for 
    创建一个名为favorite_places的字典。
    在这个字典中，将三个人的名字用作键；对于其中的每个人，都存储他喜欢的1〜3个地方. 
    朋友指出他们喜欢的一个地方(input)。遍历这个字典，并将其中每个人的名字及其喜欢的地方打印出来
'''

#创建字典
favorite_places = {'张三':['上海','北京','广州'],'李四':['九寨沟','张家界','鼓浪屿']}
# name = input('请输入名字:')
# for k in favorite_places:
#     print(k)
    # if name == k:
    #     print(favorite_places[name])

# for k in favorite_places.values():
#     print(k)

# print(favorite_places.items()) #获取到字典的 key 和value 组成的元组字典
# for k,v in favorite_places.items():
#     print(k,v)




'''
   99乘法表 
   用for循环打印99乘法表
   用while
   1x1 =1
   1x2 = 2 2x2 = 4
'''
# num = [1,2,3,4,5,6,7,8,9]
# print(list(range(1,10)))#转换为列表进行输出
# print(list(range(5)))

# for i in range(1,10):
#     # print(i)
#     for j in range(1,i+1):  #1+1 2  1
#         print('%d x %d = %d'%(j,i,(j*i)),end = '\t') #输出不换行
#     print()   #以换行为结尾

# i = 1
# while i < 10:
#     # print(i)
#     j = 1
#     while j<=i:
#         print('%d x %d = %d' % (j, i, (j * i)), end='\t')
#         j+=1
#     i+=1
#     print()

'''
    1-100的和
    1+2+3+4+...+100 = ?
    range(1,101)
'''
# j = 0
# for i in range(1,101):
#     j+=i  # j = 0+1   3
#     # print(i)
# print(j)
'''
    从键盘输入一个字符串，将小写字母全部转换成大写字母,
    将字符串以列表的形式输出(如果字符串包含整数,转为整型)?
'''
# str1 = input('请输入一个字符串:')  #接受一个字符串 dsdfa1234
# list1 = []   #生成一个空列表
# for i in str1:  #遍历输入的字符串
#     # print(i)
#     if i.isdecimal() == True:  #判断当前字符是否是数字
#         list1.append(int(i))   #将数字转换为int类型然后插入到列表当中
#     else:
#         list1.append(i.upper())#将当前字符转换为大写，然后追加到列表当中
# print(list1)

'''
    随机输入8位以内的的正整数，要求：一、求它是几位数，二、逆序打印出各位数字。
'''
#while
#判断是否是数字 if
# num = input('请输入一个8位以内数：')
# if len(num) <= 8 :
#    print('这个数是%d位数'%len(num))
#    print(num[::-1])
# else:
#     print('输入有误')

'''
    一球从n米(自己输入)高度自由落下，每次落地后反跳回原高度的一半；
    再落下，求它在第10次落地时，共经过多少米？
'''
# n = 10
# m = 0
# for i in range(1,4):
#     #第一次触地
#     if i == 1 :
#         m += n
#     else:#第二次触底是与那里高度的一半
#         n/=2
#         m+=n*2  #在累加起来
# print(m)

'''   
    输入一行字符，分别统计出其中英文字母、空格、数字和其它字符的个数。
'''
#输入字符 dsf  343
#判断是字符还是数字还是空格  统计出个数 isalpha  isdigit isspace
#输出统计的结果
# str1 = input('请输入字符：')
# zf = 0
# num = 0
# space = 0
# other = 0
# for i in str1:
#     if i.isalpha():
#         zf+=1
#     elif i.isdigit():
#         num+=1
#     elif i.isspace():
#         space+=1
#     else:
#         other+=1
#
# print('当前字母的个数是：%d,当前数字的个数是：%d,当前的空格个数是：%d,其他字符的个数是：%d'%(zf,num,space,other))




'''
names = ['Tom','Billy','Jefferson','Andrew','Wesley','Steven',
'Joe','Alice','Jill','Ana','Wendy','Jennifer','Sherry','Eva']
找出上述名字中长度大于4的名字,组成列表打印出来.
过滤掉长度大于5的字符串列表，并将剩下的转换成大写字母.
'''
names = ['Tom','Billy','Jefferson','Andrew','Wesley','Steven',
'Joe','Alice','Jill','Ana','Wendy','Jennifer','Sherry','Eva']

# name = []
# for i in names:
#     if len(i) > 4 :
#         name.append(i)
# print(name)
# name = [i for i in names if len(i) > 4] #生成一个列表
# print(name)
# name = [i.upper() for i in names if len(i) < 5]
# print(name)
'''
求M,N中矩阵对应元素的和，元素的乘积
m = [[1,2,3],   m[0][0] + n[0][0]
    [4,5,6],
    [7,8,9]] 
n = [[2,2,2],
    [3,3,3],
    [4,4,4]] 
'''
m = [[1,2,3],
    [4,5,6],
    [7,8,9]]
n = [[2,2,2],
    [3,3,3],
    [4,4,4]]

# for i in range(3):
#     for j in range(3):
#         print(m[i][j]+n[i][j])

# num = [m[i][j]+n[i][j] for i in range(3) for j in range(3)]
# print(num)
#
# num2 = [m[i][j]*n[i][j] for i in range(3) for j in range(3)]
# print(num2)



'''
打印出所有的“水仙花数”，所谓“水仙花数”是指一个三位数，
其各位数字立方和等于该数本身。
例如：153是一个“水仙花数”，因为153=1的三次方＋5的三次方＋3的三次方。
程序分析：利用for循环控制100-999个数，每个数分解出个位，十位，百位。
'''
# 153 = 1**3 + 5**3 3**3
# 遍历 100 1000
# 获取每一位数的 百位， 十位 个位
#取百位
# print(num//100)
# #取十位
# print(num%100//10)
# #取个位数
# print(num%100%10)
#分别 **3 然后相加 判断 与当前数是否相等
# for i in range(100,1000):
#     x = i//100
#     y = i%100//10
#     z = i%100%10
#     if i == (x**3)+(y**3)+(z**3) :
#         print(i)


'''
    打印菱形
'''
# i = '*'
# m = '***'
# print(i.center(7))
# print(m.center(7))

# for i in range(1,8,2):
#     print(('*'*i).center(7))
#     if i == 7:
#         for i in range(5, 0, -2):
#             print(('*' * i).center(7))
# i = '*'
# n = int(input('n = '))
# l = []
# l.append(i)
# for index in range(1,n):
#     print(i.center(n * 2 + 1))
#     i+= '**'
#     l.append(i)
# for index  in l[::-1]:
#     print(index.center(n * 2 + 1))

'''
一个5位数，判断它是不是回文数。即12321是回文数，
个位与万位相同，十位与千位相同。
'''
# num = input('请输入一个数')
# if num[0] == num[4] and num[1] == num[3]:
#     print('%s是一个回文数'%num)
# else:
#     print('%s不是一个回文数'%num)

# if num == num[::-1]:
#     print('%s是一个回文数' % num)
# else:
#     print('%s不是一个回文数' % num)



'''
    求一个3*3矩阵对角线元素之和 
    m [0][0]1    m [0][2-0=2]  3
    m [1][1]5    m [1][2-1=1]  5
    m [2][2]9    m [2][2-2=0]  7  
'''
# m = [[1,2,3],
#     [4,5,6],
#     [7,8,9]]
# x = 0
# y = 0
# for i in range(3):
#     x+=m[i][i]
#     y+=m[i][2-i]
# print(x)
# print(y)


'''
    题目：有四个数字：1、2、3、4，能组成多少个互不
    相同且无重复数字的三位数？各是多少？
    123 213 143 234 431  
    程序分析：可填在百位、十位、个位的数字都是1、2、3、4。
    组成所有的排列后再去 掉不满足条件的排列。(用列表推导式)
'''
# for x in range(1,5):
#     for y in range(1,5):
#         for z in range(1,5):
#             if x != y and x != z and y != z:
#                 print(x*100+y*10+z)

# num = [x*100+y*10+z  for x in range(1,5) for y in range(1,5) for z in range(1,5) if x != y and x != z and y != z]
# print(num)

'''
    将列表用for循环添加到另一个字典中
'''
names = ['Tom','Billy','Jefferson','Andrew','Wesley','Steven',
'Joe','Alice','Sherry','Eva']
# print(list(enumerate(names)))

# name = {k:v for k,v in enumerate(names)}
# print(name)


'''
    设一组账号和密码不少于两个
    通过输入账号和密码，如果输入正确则显示登录成功
    若账号或者密码错误则显示登录失败，最对可以输入三次
'''
# users ={'张三':'123456','李四':'654321'}
#
# for i in range(3):
#     name = input('请输入账号：')
#     pwd = input('请输入密码：')
#     if name in users:
#         if pwd == users[name]:
#             print('登录成功')
#             break
#         else:
#             if i== 2:
#                 print('您的账号已冻结！')
#                 break
#             print('您的密码有误！')
#     else:
#         if i == 2:
#             print('您的账号已冻结！')
#             break
#         print('您的账号有误！')



'''
    求阶乘 用while 和for 分别实现
    例： 2的阶乘 2*1 3的阶乘位 3*2*1   4的阶乘位4*3*2*1 
    求 5 的阶乘
'''
# num = int(input('请输入一个数:'))

#for
# x = 1
# for i in range(1,num+1):
#     x*=i
# print(x)

# x = 1
# y = 1
# while x <= num:
#     y*=x
#     x+=1
# print(y)


'''
    冒泡排序
'''
# list01 = [2,6,4,9,3,10]
# for i in range(len(list01)): # 0 1
#     for j in range(1,len(list01)-i):
#         # print(list01[j])
#         # print(list01[j-1])
#         if list01[j] < list01[j-1]:
#             list01[j],list01[j-1] = list01[j-1],list01[j]
#             # a = list01[j]
#             # list01[j] = list01[j - 1]
#             # list01[j - 1] = a
#             print(list01)
#         print('_'*50)
# [6, 9, 4, 10, 3, 2]

# list01 = [2,6,4,9,3,10]


#
# print(list01)




