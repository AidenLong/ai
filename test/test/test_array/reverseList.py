# -*- coding:utf-8 -*-

class Node:

    def __init__(self, value, next):
        self.value = value
        self.next = next


def reverseList(head):
    pre = None
    curr = head
    while curr:
        next_tmp = curr.next
        curr.next = pre
        pre = curr
        curr = next_tmp
    return pre

