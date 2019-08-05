# -*- coding:utf-8 -*-

from random import random
from math import sqrt


# circle = 0
# num = 10000
# for i in range(num):
#     x, y = random(), random()
#     dist = sqrt(x ** 2 + y ** 2)
#     if dist <= 1:
#         circle += 1
#     pi = 4 * (circle / num)

# print(pi)


'''
乘积小于K的连续子数组
'''
def numSubarrayProductLessThanK(nums, k):
    if k <= 1: return 0
    prod = 1
    ans = left = 0
    for right, val in enumerate(nums):
        prod *= val
        while prod >= k:
            prod /= nums[left]
            left += 1
        ans += right - left + 1
    return ans


print(numSubarrayProductLessThanK(nums=[10, 5, 2, 6], k=100))

'''
第二道，平面直接坐标系给一些了点对，找出其组成的最小矩形的面积，若没有矩形返回0。
如果4个点4层循环遍历要O(n^ 4)，然后面试官提示我如果2组点对的中点相同且距离相等，
那么他们可以构成一个长方形
'''


def minAreaRect(points):
    mid_point = {}
    min_area = 0
    for p1 in points:
        for p2 in points:
            if p1 != p2:
                x, y = abs(p1[0] - p2[0]) / 2, abs(p1[1] - p2[1]) / 2
                mid_point_str = str(x) + ',' + str(y)
                if mid_point_str in mid_point:
                    if min_area == 0 or mid_point[mid_point_str] < min_area:
                        if mid_point[mid_point_str] != 0:
                            min_area = mid_point[mid_point_str]
                else:
                    mid_point[mid_point_str] = abs(p1[0] - p2[0]) * abs(p1[1] - p2[1])
    return min_area


print(minAreaRect([[1, 1], [1, 3], [3, 1], [3, 3], [4, 1], [4, 3]]))

'''
用如下结构体定义一个以为数轴上的线段，起点终点分别为(start, end)

当两条线段满足如下空间关系时，我们说线段 seg1 可以完全覆盖线段 seg2
seg1：
|---------------|
seg2：
   |-----|
如线段 (1, 5) 可以完全覆盖 (2, 4)
 
在此之上进行引申
定义线段数组 S1 包含 N 条线段
线段数组 S2 包含 M 条线段
 
可否实现一个函数，用来判断线段数组 S1 是否完全覆盖 S2
'''


class Segment:

    def __init__(self, start, end):
        self.start = start
        self.end = end


def is_cover(s1, s2):
    s1_data = s1_data = [s1[0].start, s1[0].end, s1[0].start, s1[0].end]
    s2_data = [s2[0].start, s2[0].end, s2[0].start, s2[0].end]
    for s in s1:
        if s.start < s1_data[0]:
            s1_data[0] = s.start
            if s.end > s1_data[1]:
                s1_data[1] = s.end
        if s.end > s1_data[3]:
            s1_data[3] = s.end
            if s.start < s1_data[2]:
                s1_data[2] = s.start
    for s in s2:
        if s.start < s2_data[0]:
            s2_data[0] = s.start
            if s.end > s2_data[1]:
                s2_data[1] = s.end
        if s.end > s2_data[3]:
            s2_data[3] = s.end
            if s.start < s2_data[2]:
                s2_data[2] = s.start

    print(s1_data)
    print(s2_data)
    if s1_data[0] > s2_data[0] or s1_data[3] < s2_data[3]:
        return False
    else:
        if s1_data[1] > s1_data[2]:
            return True
        if s1_data[1] > s2_data[1] and (s1_data[2] < s2_data[2] or s2_data[2] < s1_data[1]):
            return True
        else:
            return False


print(is_cover([Segment(1, 1.9), Segment(2, 4)], [Segment(1.3, 1.5)]))
