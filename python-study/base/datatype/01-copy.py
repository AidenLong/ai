#_*_ conding:utf-8 _*_
#直接赋值
a = [1,2,3]
# b = a
# print(id(a))   #通过id查看变量在内存中的地址
# print(id(b))
#
# a[0] = 5     #修改的是a
# print(a)
# print(b)


#浅拷贝
# a = [1,2,3]
# b = [11,22,33]
# c = [111,222,333]
#
# list01 = [a,b,c]
# print(id(list01))
# list02 = list01[:]
# #查看list01 和 list02
# print(list01)
# print(list02)
# print('_'*20)
# #检查list01 和 list02 在内存中的地址
# print(id(list01))
# print(id(list02))
#
# #修改一下
# a[0] = 5
# # print(a)
# print(list01)
# print(list02)


#深拷贝
import copy
a = [1,2,3]
b = [11,22,33]
c = [111,222,333]

list01 = [a,b,c]
print(id(list01))

#赋值
list02 = copy.deepcopy(list01)
#查看list01 和 list02
print(list01)
print(list02)
print('_'*20)
#检查list01 和 list02 在内存中的地址
print(id(list01))   #
print(id(list02))   #

#修改一下
a[0] = 5
# print(a)
print(list01)
print(list02)





















