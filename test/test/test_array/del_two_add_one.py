# -*- coding:utf-8 -*-


def get_max_two(a, b, array):
    if len(array) == 1 and array[0] > b:
        return a, array[0], True, False
    if len(array) >= 2:
        max_value = max(array)
        if max_value > a:
            a = max_value
            array.remove(max_value)
        max_value = max(array)
        if max_value > b:
            b = max_value
            array.remove(max_value)
        return a, b, True, True
    return a, b, False, False


def del_two_add_one(array):
    if not array:
        return 0
    if len(array) == 1:
        return array[0]

    array.sort()
    need_add = []
    while len(array) + len(need_add) >= 2:
        if len(array) < 2:
            a, b, flag_1, flag_2 = get_max_two(array[0] if len(array) == 1 else 0, 0, need_add)
        else:
            a, b, flag_1, flag_2 = get_max_two(array[-1], array[-2], need_add)
        add_value = abs(a - b)
        if not flag_1:
            array.remove(b)
        if not flag_2:
            array.remove(a)
        need_add.append(add_value)

    return array[0] if len(array) == 1 else need_add[0]


if __name__ == '__main__':
    print(del_two_add_one([1, 3, 7, 2]))
