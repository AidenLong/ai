# -*-encoding=utf8-*-
import csv
import jieba
import jieba.posseg as pseg
import os

fout2 = open('example_test.test', 'w', encoding='utf8')
dics = csv.reader(open("DICT_NOW.csv", 'r', encoding='utf8'))
fuhao = ['；', '。', '?', '？', '!', '！', ';']
biaoji = ['DIS', 'SYM', 'SGN', 'TES', 'DRU', 'SUR', 'PRE', 'PT', 'Dur', 'TP', 'REG', 'ORG', 'AT', 'PSB', 'DEG', 'FW',
          'CL']

for row in dics:
    if len(row) == 2:
        jieba.add_word(row[0].strip(), tag=row[1].strip())

files = os.listdir("D:\syl\\ai\python-workspace\\ai\\nlp\\NERuselocal\\cyqk\\")
path_dir = "D:\syl\\ai\python-workspace\\ai\\nlp\\NERuselocal\\cyqk\\"

split_num = 0
for file in files:
    if split_num >= 10:
        break
    if "txtoriginal" in file:
        fp = open(path_dir + file, 'r', encoding='utf8')
        lines = [line for line in fp]
        for line in lines:
            split_num += 1
            words = pseg.cut(line)
            for key, value in words:
                print(key)
                print(value)
                if value.strip() and key.strip():

                    if value not in biaoji:
                        value = 'O'
                        for achar in key.strip():
                            if achar and achar.strip() in fuhao:
                                string = achar + " " + value.strip() + "\n" + "\n"
                                fout2.write(string)

                            elif achar.strip() and achar.strip() not in fuhao:
                                string = achar + " " + value.strip() + "\n"
                                fout2.write(string)
                            else:
                                continue

                    elif value.strip() in biaoji:
                        begin = True
                        for char in key.strip():
                            if begin:
                                begin = False
                                string1 = char + ' ' + 'B-' + value.strip() + '\n'
                                fout2.write(string1)

                            else:
                                string1 = char + ' ' + 'I-' + value.strip() + '\n'

                                fout2.write(string1)

                    else:
                        continue
fout2.close()
