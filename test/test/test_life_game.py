# -*- coding:utf-8 -*-

import timeit
import numpy as np

Znp = np.random.randint(2, size=(1000, 1000))


def np_solver(Z):
    N = (Z[0:-2, 0:-2] + Z[0:-2, 1:-1] + Z[0:-2, 2:] +
         Z[1:-1, 0:-2] + Z[1:-1, 2:] +
         Z[2:, 0:-2] + Z[2:, 1:-1] + Z[2:, 2:])

    # 繁衍规则
    birth = (N == 3) & (Z[1:-1, 1:-1] == 0)
    # 若本来为活，且邻居数为2或者3，则持续为活
    survive = ((N == 2) | (N == 3)) & (Z[1:-1, 1:-1] == 1)
    # 全部重置为死亡
    Z[...] = 0
    Z[1:-1, 1:-1][birth | survive] = 1
    return Z  # 把实现填进来


print(timeit.timeit(lambda: np_solver(Znp), number=3))
0