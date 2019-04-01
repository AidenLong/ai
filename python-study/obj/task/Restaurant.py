#-*- conding:utf-8 -*-
'''
1.餐馆：创建一个名为Restaurant的类，其方法__init__()设置两个属性：
restaurant_name 和 cuisine_type(烹饪)。
创建一个名为 describe_restaurant()方法和一个名为open_restaurant ()
方法,其中前者打印前述两项信息，而后者打印一条消息,指出餐馆正在营业。
根据这个类创建一个名为restaurant的实例，分别打印其两个属性，
再调用前述两个方法。

2.三家餐馆：根据你为完成练习1而编写的类创建三个实例，
并对每个实例调用方法 describe_restaurant()。

4.就餐人数：在为完成练习1而编写的程序中,添加一个名为number_served的属性,
并将其默认值设置为0。打印有多少人在这家餐馆就餐过,然后修改这个值并再次打印
它。
添加一个名为set_number_served()的方法,它让你能够设置就餐人数。
调用这个方法并向它传递一个值，然后再次打印这个值。
添加一个名为increment_number_served()的方法,它让你能够将就餐人数递增.
调用这个方法并向它传递一个这样的值：你认为这家餐馆每天可能接待的就餐人数。


'''
class Restaurant():
    '''
        餐馆类
    '''
    def __init__(self,restaurant_name,cuisine_type,number_served = 0):
        #声明两个实例属性
        self.restaurant_name = restaurant_name #餐馆名字
        self.cuisine_type = cuisine_type       #菜系
        self.number_served = number_served

    def describe_restaurant(self):
        print('名称：%s,菜系：%s'%(self.restaurant_name,self.cuisine_type))


    def open_restaurant(self):
        print('欢迎光临%s,正在营业'%self.restaurant_name)


    #设置就餐人数
    def set_number_served(self,n):
        self.number_served = n  #通过传递的参数给实例属性赋值
        print('当前就餐人数:%d'%self.number_served)

    #递增增加就餐人数
    def increment_number_served(self,n):
        for i in range(1,n+1):
            self.number_served += 1
            print('当前就餐人数:%d'%self.number_served)



if __name__ == '__main__':
    #练习1.
    # restaurant = Restaurant('金拱门','西餐')
    #打印两个属性
    # print(restaurant.restaurant_name)
    # print(restaurant.cuisine_type)

    #调用两个方法
    # restaurant.describe_restaurant()
    # restaurant.open_restaurant()

    # 练习2. 创建三个实例
    # chuancai = Restaurant('我家酸菜鱼', '川菜')
    # chuancai.describe_restaurant()
    #
    # xiangcai = Restaurant('沪上湘城', '湘菜')
    # xiangcai.describe_restaurant()
    #
    # loulan = Restaurant('楼兰', '新疆菜')
    # loulan.describe_restaurant()

    #4.
    loulan = Restaurant('楼兰', '新疆菜')
    # print('就餐人数:%d'%loulan.number_served)
    # loulan.number_served = 10  #通过对象名.属性名设置 属性值
    # print('就餐人数:%d' % loulan.number_served)
    loulan.set_number_served(40)
    loulan.increment_number_served(10)

