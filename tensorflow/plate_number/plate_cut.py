# -- encoding:utf-8 --
"""
提取车牌号(使用OpenCV的方式)
Create by ibf on 2018/5/28
"""

import cv2
import numpy as np


def preprocess(gray):
    """
    对灰度对象进行形态转换（预处理）
    :param gray:
    :return:
    """
    # 高斯平滑
    gaussian = cv2.GaussianBlur(gray, (3, 3), 0, 0, cv2.BORDER_DEFAULT)

    # 中值滤波
    median = cv2.medianBlur(gaussian, 5)

    # Sobel算子，对边缘进行处理(获取边缘信息，其实就是卷积过程)
    # x：[-1, 0, +1, -2, 0, +2, -1, 0, +1]
    # y：[-1, -2, -1, 0, 0, 0, +1, +2, +1]
    sobel = cv2.Sobel(median, cv2.CV_8U, 1, 0, ksize=3)

    # 二值化
    ret, binary = cv2.threshold(sobel, 170, 255, cv2.THRESH_BINARY)

    # 膨胀&腐蚀
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 7))
    # 膨胀一次，让轮廓突出
    dilation = cv2.dilate(binary, element2, iterations=1)
    # 腐蚀一次，去掉细节
    erosion = cv2.erode(dilation, element1, iterations=1)
    # 再次膨胀，让轮廓明显一些
    dilation2 = cv2.dilate(erosion, element2, iterations=3)

    # cv2.imshow('ss', dilation2)
    # cv2.waitKey(0)
    return dilation2


def find_plate_number_region(img):
    """
    寻找可能是车牌区域的轮廓
    :param img:
    :return:
    """
    # 查找轮廓(img: 原始图像，contours：矩形坐标点，hierarchy：图像层次)
    img, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 查找矩形
    max_ratio = -1
    max_box = None
    ratios = []
    number = 0
    for i in range(len(contours)):
        cnt = contours[i]

        # 计算轮廓面积
        area = cv2.contourArea(cnt)
        # 面积太小的过滤掉
        if area < 1000:
            continue

        # 找到最小的矩形
        rect = cv2.minAreaRect(cnt)

        # 矩形的四个坐标（顺序不定，但是一定是一个左下角、左上角、右上角、右下角这种循环顺序(开始是哪个点未知)）
        box = cv2.boxPoints(rect)
        # 转换为long类型
        box = np.int0(box)

        # 计算长宽高
        height = abs(box[0][1] - box[2][1])
        weight = abs(box[0][0] - box[2][0])
        ratio = float(weight) / float(height)
        # 正常的车牌宽高比在2.7~5之间
        if ratio > max_ratio:
            max_box = box

        if ratio > 5.5 or ratio < 2:
            continue

        # 将结果添加到序列中
        number += 1
        ratios.append((box, ratio))

    # 根据找到的图像矩阵数量进行数据输出
    if number == 1:
        # 直接返回
        return ratios[0][0]
    elif number > 1:
        # 如果存在多个，这里不做太多考虑，直接返回一个最可能的（也就是ratio在2.7~5之间的中间数字的那个图像）
        # TODO: 实际上可以在这里训练一个模型，用于判断图片中是否有车牌照（就一个简单的神经网络即可）
        filter_ratios = list(filter(lambda t: t[1] >= 2.7 and t[1] <= 5, ratios))
        size_filter_ratios = len(filter_ratios)
        if size_filter_ratios == 1:
            return filter_ratios[0][1]
        elif size_filter_ratios > 1:
            r = [filter_ratios[i][1] for i in range(size_filter_ratios)]
            sort_r = np.argsort(r)
            return filter_ratios[int(len(sort_r) / 2)][0]
        else:
            # 直接返回最大值
            max_index = np.argmax(ratios, 0)
            return ratios[max_index[1]][1]
    else:
        # 直接返回最大值
        return max_box


def cut(img_or_img_path):
    """
    截取车牌区域
    :param img:
    :return:
    """
    if isinstance(img_or_img_path, str):
        img = cv2.imread(img_or_img_path)
    else:
        img = img_or_img_path

    # 将图像转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 图像预处理
    dilation = preprocess(gray)

    # 查找车牌区域(只会有一个)
    box = find_plate_number_region(dilation)

    # 返回区域对应的图像
    # 因为不知道，点的顺序，所以对左边点坐标排序
    ys = [box[0, 1], box[1, 1], box[2, 1], box[3, 1]]
    xs = [box[0, 0], box[1, 0], box[2, 0], box[3, 0]]
    ys_sorted_index = np.argsort(ys)
    xs_sorted_index = np.argsort(xs)

    # 获取x上的坐标
    x1 = box[xs_sorted_index[0], 0]
    x2 = box[xs_sorted_index[3], 0]

    # 获取y上的坐标
    y1 = box[ys_sorted_index[0], 1]
    y2 = box[ys_sorted_index[3], 1]

    # 截取图像
    img_plate = img[y1:y2, x1:x2]

    return img_plate


if __name__ == '__main__':
    path = '0.jpg'
    cut_img = cut(path)
    print(cut_img.shape)
    cv2.imwrite('test.jpg', cut_img)
