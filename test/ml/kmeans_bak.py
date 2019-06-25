# -*- coding:utf-8 -*-
import numpy as np
import random
import matplotlib.pyplot as plt


def calcu_distance(vec1, vec2):
    # 计算向量vec1和向量vec2之间的欧式距离
    return np.sqrt(np.sum(np.square(vec1 - vec2)))


def load_data_set(in_file):
    with open(in_file, 'r') as file:
        data = file.read()
    data_set = list()
    for line in data.split('\n'):
        attr = line.strip().split('\t')
        data = [float(i) for i in attr]
        data_set.append(data)
    return data_set


def init_centroids(data_set, k):
    return random.sample(data_set, k)  # 从data_set中随机获取k个数据项返回


def min_distance(data_set, centroid_list):
    # 对每个属于dataSet的item，计算item与centroidList中k个质心的欧式距离，找出距离最小的，
    # 并将item加入相应的簇类中

    clusterDict = dict()  # 用dict来保存簇类结果
    for item in data_set:
        vec1 = np.array(item)  # 转换成array形式
        flag = 0  # 簇分类标记，记录与相应簇距离最近的那个簇
        minDis = float("inf")  # 初始化为最大值

        for i in range(len(centroid_list)):
            vec2 = np.array(centroid_list[i])
            distance = calcu_distance(vec1, vec2)  # 计算相应的欧式距离
            if distance < minDis:
                minDis = distance
                flag = i  # 循环结束时，flag保存的是与当前item距离最近的那个簇标记

        if flag not in clusterDict.keys():  # 簇标记不存在，进行初始化
            clusterDict[flag] = list()
        # print flag, item
        clusterDict[flag].append(item)  # 加入相应的类别中

    return clusterDict  # 返回新的聚类结果


def get_centroids(cluster_dict):
    # 得到k个质心
    centroids_list = list()
    for key in cluster_dict.keys():
        centroid = np.mean(np.array(cluster_dict[key]), axis=0)
        centroids_list.append(centroid)
    return centroids_list


def get_var(cluster_dict, centroid_list):
    # 计算簇集合的均方误差
    # 将簇类中各个向量与质心的距离进行累加求和

    sum = 0.0
    for key in cluster_dict.keys():
        vec1 = np.array(centroid_list[key])
        distance = 0.0
        for item in cluster_dict[key]:
            vec2 = np.array(item)
            distance += calcu_distance(vec1, vec2)
        sum += distance
    return sum


def show_cluster(cluster_dict, centroid_list):
    # 展示聚类结果
    colorMark = ['or', 'ob', 'og', 'ok', 'oy', 'ow']  # 不同簇类的标记 'or' --> 'o'代表圆，'r'代表red，'b':blue
    centroidMark = ['dr', 'db', 'dg', 'dk', 'dy', 'dw']  # 质心标记 同上'd'代表棱形
    for key in cluster_dict.keys():
        plt.plot(centroid_list[key][0], centroid_list[key][1], centroidMark[key], markersize=12)  # 画质心点
        for item in cluster_dict[key]:
            plt.plot(item[0], item[1], colorMark[key])  # 画簇类下的点

    plt.show()


if __name__ == '__main__':
    in_file = 'test_data'
    dataSet = load_data_set(in_file)  # 载入数据集
    centroidList = init_centroids(dataSet, 4)  # 初始化质心，设置k=4
    clusterDict = min_distance(dataSet, centroidList)  # 第一次聚类迭代
    newVar = get_var(clusterDict, centroidList)  # 获得均方误差值，通过新旧均方误差来获得迭代终止条件
    oldVar = -0.0001  # 旧均方误差值初始化为-1
    print('***** 第1次迭代 *****')

    print('簇类')
    for key in clusterDict.keys():
        print(key, ' --> ', clusterDict[key])
    print('k个均值向量: ', centroidList)
    print('平均均方误差: ', newVar)
    show_cluster(clusterDict, centroidList)  # 展示聚类结果

    k = 2
    while abs(newVar - oldVar) >= 0.0001:  # 当连续两次聚类结果小于0.0001时，迭代结束
        centroidList = get_centroids(clusterDict)  # 获得新的质心
        clusterDict = min_distance(dataSet, centroidList)  # 新的聚类结果
        oldVar = newVar
        newVar = get_var(clusterDict, centroidList)

        print('***** 第%d次迭代 *****' % k)
        print('簇类')
        for key in clusterDict.keys():
            print(key, ' --> ', clusterDict[key])
        print('k个均值向量: ', centroidList)
        print('平均均方误差: ', newVar)
        show_cluster(clusterDict, centroidList)  # 展示聚类结果

        k += 1
