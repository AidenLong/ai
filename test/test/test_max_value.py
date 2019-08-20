# -*- coding:utf-8 -*-

def max_value(array):
    result_value = 0
    buy_in = False
    last_buy_in_value = 0

    for i in range(len(array) - 1):
        if not buy_in:
            if i == 0:
                if array[i] < array[i + 1]:
                    result_value -= array[i]
                    last_buy_in_value = array[i]
                    buy_in = True
                    continue
            else:
                if array[i - 1] > array[i] and array[i] < array[i + 1]:
                    result_value -= array[i]
                    last_buy_in_value = array[i]
                    buy_in = True
                    continue

        if buy_in:
            if array[i] > array[i + 1]:
                result_value += array[i]
                buy_in = False
    if buy_in:
        if array[-1] > last_buy_in_value:
            result_value += array[-1]
        else:
            result_value += last_buy_in_value

    return result_value


if __name__ == '__main__':
    print(max_value([10, 9, 8, 20]))
