# -*- coding:utf-8 -*-

def a():
    print('a方法')

print(__name__)  # 在当前脚本输出__main__
# 只允许在当前脚本执行
if __name__ == '__main__':
    print(__name__)  # 在当前脚本输出__main__
    a()
