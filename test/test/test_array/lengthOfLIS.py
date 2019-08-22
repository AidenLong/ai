# -*- coding:utf-8 -*-

def lengthOfLIS(nums):
    if len(nums) == 0:
        return 0
    dp = [1] * len(nums)
    maxans = 1
    for i in range(1, len(nums)):
        maxval = 0
        for j in range(i):
            if nums[i] > nums[j]:
                maxval = max(maxval, dp[j])
        dp[i] = maxval + 1
        maxans = max(maxans, dp[i])

    return maxans


if __name__ == '__main__':
    print(lengthOfLIS([10, 9, 2, 5, 3, 7, 101, 18]))
