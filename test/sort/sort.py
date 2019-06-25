# -*- coding:utf-8 -*-

# 冒泡排序
def maopao(array):
    for i in range(len(array)):
        do_sorted = False
        for j in range(len(array) - i - 1):
            if array[j] > array[j + 1]:
                array[j], array[j + 1] = array[j + 1], array[j]
                do_sorted = True
        if not do_sorted:
            break
    return array


# 一次快排
def partition1(array, left, right):
    tmp = array[left]
    while left < right:
        while left < right and array[right] >= tmp:
            right -= 1
        array[left] = array[right]
        while left < right and array[left] <= tmp:
            left += 1
        array[right] = array[left]
    array[left] = tmp
    return left


def partition(array, left, right):
    tmp = array[left]
    while left < right:
        while left < right and array[right] >= tmp:
            right -= 1
        array[left], array[right] = array[right], array[left]
        while left < right and array[left] <= tmp:
            left += 1
        array[left], array[right] = array[right], array[left]
    return left


# 快速排序
def quick_sort(array, left, right):
    if left < right:
        mid = partition(array, left, right)
        quick_sort(array, left, mid - 1)
        quick_sort(array, mid + 1, right)


# 一次归并
def merge(array, low, mid, high):
    """
    两个需要归并排序的序列，从左向右遍历，逐一比较，小的放到tmp中
    :param array:
    :param low:
    :param mid:
    :param high:
    :return:
    """
    tmp = []
    i = low
    j = mid + 1
    while i <= mid and i <= high:
        if array[i] <= array[j]:
            tmp.append(array[i])
            i += 1
        else:
            tmp.append(array[j])
            j += 1
    while i <= mid:
        tmp.append(array[i])
        i += 1
    while j <= high:
        tmp.append(array[j])
        j += 1
    array[low: high + 1] = tmp


def merge_sort(array, low, high):
    if low < high:
        mid = (low + high) // 2
        merge_sort(array, low, mid)
        merge_sort(array, mid + 1, high)
        merge(array, low, mid, high)


if __name__ == '__main__':
    data = [1, 3, 5, 2, 4, 6]
    # print(maopao(data))
    quick_sort(data, 0, 5)
    print(data)
    # data = [1, 3, 5, 2, 4, 6]
    # merge_sort(data, 0, 5)
    # print(data)
