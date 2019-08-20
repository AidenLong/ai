# -*- coding:utf-8 -*-

"""
搜索螺旋排序数组(不重复)[6, 7, 0, 1, 2, 4, 5]
时间复杂度为 O(log(N))
"""


def search_none_repeat(nums, target):
    left = 0
    right = len(nums)
    while left < right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        if nums[mid] >= nums[left]:
            if nums[left] <= target < nums[mid]:
                right = mid
            else:
                left = mid + 1
        else:
            if nums[mid] < target <= nums[right - 1]:
                left = mid + 1
            else:
                right = mid
    return -1


"""
搜索螺旋排序数组(可重复)[6, 7, 0, 0, 1, 2, 4, 5]
"""


def search_repeat(nums, target):
    left = 0
    right = len(nums) - 1
    while left < right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return True
        if nums[mid] > nums[left]:
            if nums[left] <= target < nums[mid]:
                right = mid
            else:
                left = mid + 1
        elif nums[mid] < nums[left]:
            if nums[mid] < target <= nums[right - 1]:
                left = mid + 1
            else:
                right = mid
        else:
            left += 1
    return False


if __name__ == '__main__':
    print(search_none_repeat([6, 7, 0, 1, 2, 4, 5], 0))
    print(search_repeat([6, 7, 0, 0, 1, 2, 4, 5], 3))
