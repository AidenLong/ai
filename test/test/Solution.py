# -*- coding:utf-8 -*-

"""
给定一个二叉树，返回其按层次遍历的节点值。 （即逐层地，从左到右访问所有节点）。
"""


# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


def level_order(root):
    """
    :type root: TreeNode
    :rtype: List[List[int]]
    """
    queue = [root]
    res = []
    if not root:
        return []
    while queue:
        templist = []
        templen = len(queue)
        for i in range(templen):
            temp = queue.pop(0)
            templist.append(temp.val)
            if temp.left:
                queue.append(temp.left)
            if temp.right:
                queue.append(temp.right)
        res.append(templist)
    return res


if __name__ == '__main__':
    root_node = TreeNode(1)
    root_node.left = TreeNode(2)
    right = TreeNode(3)
    right.left = TreeNode(4)
    right.right = TreeNode(5)
    root_node.right = right
    print(level_order(root_node))
