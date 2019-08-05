# -*- coding:utf-8 -*-

class Solution:
    # array 二维列表
    def Find(self, target, array):
        # write code here
        rows = len(array)
        cols = len(array[0])
        if rows > 0 and cols > 0:
            row = 0
            col = cols - 1
            while row < rows and col >= 0:
                if target == array[row][col]:
                    return True
                elif target < array[row][col]:
                    col -= 1
                else:
                    row += 1
        return False


if __name__ == '__main__':
    target = 15
    array = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 12, 13]]
    answer = Solution()
    print(answer.Find(target, array))
