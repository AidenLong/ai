#_*_ conding:utf-8 _*_
'''
    创建字典

'''
# dict01 = {'name':'joe','age':18,'address':'上海'}
# print(dict01)


'''
    访问字典
'''
dict01 = {'name':'joe','age':18,'address':'上海'}
print(dict01)
print(dict01['name'])
#修改字典元素  通过找到指定 的 KEY 进行修改
# dict01['name'] = 'jack'
# print(dict01)
#增加元素
# dict01['hobby'] = '足球'
# print(dict01)

'''
    删除字典
'''
#del
# del dict01['address']
# del dict01   #从内存中删除
# print(dict01)

# dict01.clear()  #清空字典中的元素
# print(dict01)

# dict01.pop('hobby')
# print(dict01)
# print('='*50)

'''
    字典函数
'''
#str
# str1 = str(dict01)
# print(type(str1))

# dict02 = {'name':'joe','age':18,'address':'上海','sex':'女'}
# print(dict02.get('sex','男'))
#如果字典用右该key对应的元素 就输出原来的 ， 如果没有则输出你指定的

# print(dict02.keys())
# print(dict02.values())
# print(dict02.items())


#定义一个字典
# aa = {'张三':['a','b','c'],'李四':['d','e','f']}
# name = input('请输入姓名：')
# print(aa[name])


# animal1 = {'animal_name':'lala1','animal_type':'dog','animal_person':'joe'}
# animal2 = {'animal_name':'lala2','animal_type':'cat','animal_person':'susan'}
# animal3 = {'animal_name':'lala3','animal_type':'dog','animal_person':'black'}
#
# ani = [animal1,animal2,animal3]
# ani = list(ani[0].items())
# print(ani[0][0]+':'+ani[0][1])
# print(ani[1][0]+':'+ani[1][1])
# print(ani[2][0]+':'+ani[2][1])







import copy
list_0 = ["A", "B", ["C", "D"], "E"]
list_1 = copy.copy(list_0)
list_2 = list_0.copy()
list_3 = list_0[:]
list_4 = list(list_0)

# --- 深拷贝的拷贝方式 ---
list_d = copy.deepcopy(list_0)


# --- 深浅拷贝的区别 ---
# 1. 对第一层数据进行赋值
list_0[0] = "X0"
list_1[0] = "X1"
list_2[0] = "X2"
list_3[0] = "X3"
list_4[0] = "X4"
list_d[0] = "Xd"

# 打印结果: 理所当然,所有列表都发生了变化
# list_0: ['X0', 'B', ['C', 'D'], E]
# list_1: ['X1', 'B', ['C', 'D'], E]
# list_2: ['X2', 'B', ['C', 'D'], E]
# list_3: ['X3', 'B', ['C', 'D'], E]
# list_4: ['X4', 'B', ['C', 'D'], E]
# list_d: ['Xd', 'B', ['C', 'D'], E]

# 2. 对第二层的list引用进行赋值
list_0[2][0] = "Y0"
list_1[2][0] = "Y1"
list_2[2][0] = "Y2"
list_3[2][0] = "Y3"
list_4[2][0] = "Y4"
list_d[2][0] = "Yd"


# 打印结果: 0-1都被改成了同一个值,这说明浅拷贝只拷贝了第二层list的引用;而深拷贝则拷贝了数据结构
# list_0: ['X0', 'B', ['Y4', 'D'], E]
# list_1: ['X1', 'B', ['Y4', 'D'], E]
# list_2: ['X2', 'B', ['Y4', 'D'], E]
# list_3: ['X3', 'B', ['Y4', 'D'], E]
# list_4: ['X4', 'B', ['Y4', 'D'], E]
# list_d: ['Xd', 'B', ['Yd', 'D'], E]

# 3. 对第三层的Ls对象引用进行赋值
list_0[3]= "Z0"
list_1[3]= "Z1"
list_2[3]= "Z2"
list_3[3]= "Z3"
list_4[3]= "Z4"
list_d[3]= "Zd"

print(list_0)
print(list_1)
print(list_2)
print(list_3)
print(list_4)
print(list_d)

# 执行结果: 继续验证了上方论点
# list_0: ['X0', 'B', ['Y4', 'D'], Z4]
# list_1: ['X1', 'B', ['Y4', 'D'], Z4]
# list_2: ['X2', 'B', ['Y4', 'D'], Z4]
# list_3: ['X3', 'B', ['Y4', 'D'], Z4]
# list_4: ['X4', 'B', ['Y4', 'D'], Z4]
# list_d: ['Xd', 'B', ['Yd', 'D'], Zd]
