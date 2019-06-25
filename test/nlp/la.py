# -*- coding:utf-8 -*-

def ld(str1, str2):
    m, n = len(str1) + 1, len(str2) + 1

    # 初始化矩阵 m * n的矩阵
    matrix = [[0] * n for i in range(m)]
    matrix[0][0] = 0
    # 初始化第一列
    for i in range(1, m):
        matrix[i][0] = matrix[i - 1][0] + 1
    # 初始化第一行
    for j in range(1, n):
        matrix[0][j] = matrix[0][j - 1] + 1
    # 动态规划计算ld值
    for i in range(1, m):
        for j in range(1, n):
            # 如果相等，则ld值为左上角的值
            if str1[i - 1] == str2[j - 1]:
                matrix[i][j] = matrix[i - 1][j - 1]
            # 不相等时，ld值为左、左上、上中最小值加1
            else:
                matrix[i][j] = min(matrix[i - 1][j - 1], matrix[i - 1][j], matrix[i][j - 1]) + 1
    # 返回左下角的值，即为编辑距离
    return matrix[m - 1][n - 1]


if __name__ == '__main__':
    str1 = 'eeba'
    str2 = 'abac'
    print(ld(str1, str2))
