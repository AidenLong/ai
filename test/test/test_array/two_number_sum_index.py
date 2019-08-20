# -*- coding:utf-8 -*-

def twoSum(nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: List[int]
    """
    int_index = {}
    for i in range(len(nums)):
        sub = target - nums[i]
        if sub in int_index.keys():
            return [int_index[sub], i]
        else:
            int_index[nums[i]] = i


if __name__ == '__main__':
    print(twoSum([2, 7, 3, 4, 8], 9))
