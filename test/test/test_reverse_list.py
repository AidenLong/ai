# -*- coding:utf-8 -*-

class head:

    def __init__(self, value):
        self.value = value
        self.next = None


def reverseList(head):
    if not head:
        return head

    pre = None
    cur = head
    lat = head.next

    while lat:
        cur.next = pre
        pre = cur
        cur = lat
        lat = lat.next

    cur.next = pre
    return cur
