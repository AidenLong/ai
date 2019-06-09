# -*-encoding=utf8-*-
import csv
import jieba
import jieba.posseg as pseg
import re, os

fout1 = open('example.dev', 'w', encoding='utf8')
fout2 = open('example.test', 'w', encoding='utf8')
fout3 = open('example.train', 'w', encoding='utf8')
dics = csv.reader(open("DICT_NOW.csv", 'r', encoding='utf8'))
flag = 0
fuhao = ['；', '。', '?', '？', '!', '！', ';']
biaoji = ['DIS', 'SYM', 'SGN', 'TES', 'DRU', 'SUR', 'PRE', 'PT', 'Dur', 'TP', 'REG', 'ORG', 'AT', 'PSB', 'DEG', 'FW',
          'CL']

for row in dics:
    if flag == 0:
        flag = 1
        continue
    if len(row) == 2:
        jieba.add_word(row[0].strip(), tag=row[1].strip())
        # jieba.suggest_freq(segment)

files = os.listdir("D:\syl\\ai\python-workspace\\ai\\nlp\\NERuselocal\\cyqk\\")
path_dir = "D:\syl\\ai\python-workspace\\ai\\nlp\\NERuselocal\\cyqk\\"
# split_num1=sum([len(open(path_dir+file,'r',encoding='utf8').readlines()) for file in files if "txtoriginal" in file])
# if split_num%15==0:
# index=str(1)
# elif split_num%15>0 and split_num%15<4:
# index=str(2)
# else:
# index=str(3)
split_num = 0
for file in files:
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

                    if split_num % 15 < 2:
                        index = str(1)
                    elif split_num % 15 > 1 and split_num % 15 < 4:
                        index = str(2)
                    else:
                        index = str(3)

                    if value not in biaoji:
                        value = 'O'
                        for achar in key.strip():
                            if achar and achar.strip() in fuhao:
                                string = achar + " " + value.strip() + "\n" + "\n"
                                if index == '1':
                                    fout1.write(string)
                                elif index == '2':
                                    fout2.write(string)
                                elif index == '3':
                                    fout3.write(string)
                                else:
                                    pass
                            elif achar.strip() and achar.strip() not in fuhao:
                                string = achar + " " + value.strip() + "\n"
                                if index == '1':
                                    fout1.write(string)
                                elif index == '2':
                                    fout2.write(string)
                                elif index == '3':
                                    fout3.write(string)
                                else:
                                    pass
                            else:
                                continue

                    elif value.strip() in biaoji:
                        begin = True
                        for char in key.strip():
                            if begin:
                                begin = False
                                string1 = char + ' ' + 'B-' + value.strip() + '\n'
                                if index == '1':
                                    fout1.write(string1)
                                elif index == '2':
                                    fout2.write(string1)
                                elif index == '3':
                                    fout3.write(string1)
                                else:
                                    pass
                            else:
                                string1 = char + ' ' + 'I-' + value.strip() + '\n'
                                if index == '1':
                                    fout1.write(string1)
                                elif index == '2':
                                    fout2.write(string1)
                                elif index == '3':
                                    fout3.write(string1)
                                else:
                                    pass
                    else:
                        continue
fout1.close()
fout2.close()
fout3.close()
# if __name__=='__main__':
# for k,v in pseg.cut("今天中午吃什么"):
# print(k)
# print(v)
