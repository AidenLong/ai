# -*- coding:utf-8 -*-
import re


def handle_repeat_word(outputs):
    patten1 = u"(?i)\\b([\u4e00-\u9fa5_a-zA-Z0-9，' ']+)\\b(?:\\s+\\1\\b)+"
    patten2 = u"(?i)\\b([a-z]+' ')\\b(?:\\s+\\1\\b)+"
    p = re.compile(patten1)
    match = p.findall(outputs)
    if not match:
        p = re.compile(patten2)
        match = p.findall(outputs)
    if match:
        for str in match:
            outputs = outputs.replace(str, '', outputs.count(str) - 1)
    return outputs.replace(' ', '')


if __name__ == '__main__':
    str = '你 好 啊 好 啊 吃 吃 吃 a a b b b b'
    print(handle_repeat_word(str))
