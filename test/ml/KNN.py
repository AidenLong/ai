# -*- coding:utf-8 -*-
import operator as opt
import numpy as np


def norm_data(dataset):
    max_vals = dataset.max(axis=0)
    min_vals = dataset.min(axis=0)
    ranges = max_vals - min_vals
    ret_data = (dataset - min_vals) / ranges
    return ret_data, ranges, min_vals


def kNN(dataset, labels, test_data, k):
    dist_square_mat = (dataset - test_data) ** 2  # 计算差值的平方
    dist_square_sums = dist_square_mat.sum(axis=1)  # 求每一行的差值平方和
    distances = dist_square_sums ** 0.5  # 开根号，得出每个样本到测试点的距离
    sorted_indices = distances.argsort()  # 排序，得到排序后的下标
    indices = sorted_indices[:k]  # 取最小的k个
    label_count = {}  # 存储每个label的出现次数
    for i in indices:
        label = labels[i]
        label_count[label] = label_count.get(label, 0) + 1  # 次数加一
    sorted_count = sorted(label_count.items(), key=opt.itemgetter(1), reverse=True)  # 对label出现的次数从大到小进行排序
    return sorted_count[0][0]


if __name__ == '__main__':
    dataSet = np.array([[2, 3], [2.2, 3.3], [6, 8]])
    normDataSet, ranges, min_vals = norm_data(dataSet)
    labels = ['a', 'a', 'b']
    testData = np.array([3.9, 5.5])
    normTestData = (testData - min_vals) / ranges
    result = kNN(normDataSet, labels, normTestData, 2)
    print(result)
