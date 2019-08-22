# -*- coding:utf-8 -*-

class Node:
    def __init__(self, value=None, left=None, right=None):
        self.value = value
        self.left = left  # 左子树
        self.right = right  # 右子树


def preTraverse(root):
    '''
    前序遍历
    '''
    if root == None:
        return
    print(root.value)
    preTraverse(root.left)
    preTraverse(root.right)


def perorderTraverse(root):
    stack = []
    sol = []
    curr = root
    while stack or curr:
        if curr:
            sol.append(curr.value)
            stack.append(curr.right)
            curr = curr.left
        else:
            curr = stack.pop()
    return sol


def midTraverse(root):
    '''
    中序遍历
    '''
    if root == None:
        return
    midTraverse(root.left)
    print(root.value)
    midTraverse(root.right)


def inorderTraverse(root):
    stack = []
    sol = []
    curr = root
    while stack or curr:
        if curr:
            stack.append(curr)
            curr = curr.left
        else:
            curr = stack.pop()
            sol.append(curr.value)
            curr = curr.right
    return sol


def afterTraverse(root):
    '''
    后序遍历
    '''
    if root == None:
        return
    afterTraverse(root.left)
    afterTraverse(root.right)
    print(root.value)


def postorderTraverse(root):
    stack = []
    sol = []
    curr = root
    while stack or curr:
        if curr:
            sol.append(curr.value)
            stack.append(curr.left)
            curr = curr.right
        else:
            curr = stack.pop()
    return sol[::-1]


def levelOrder(root):
    if not root:
        return
    nodes = [root]
    length_nodes = len(nodes)
    sol = []
    while length_nodes > 0:
        node = nodes.pop(0)
        if node:
            sol.append(node.value)
            nodes.append(node.left)
            nodes.append(node.right)
        length_nodes = len(nodes)
    return sol


if __name__ == '__main__':
    root = Node('D', Node('B', Node('A'), Node('C')), Node('E', right=Node('G', Node('F'))))
    print('前序遍历：')
    preTraverse(root)
    print(perorderTraverse(root))
    print('\n')
    print('中序遍历：')
    midTraverse(root)
    print(inorderTraverse(root))
    print('\n')
    print('后序遍历：')
    afterTraverse(root)
    print(postorderTraverse(root))
    print('\n')
    print(levelOrder(root))
