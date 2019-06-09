# -*-encoding=utf8-*-
import csv
import jieba
import jieba.posseg as pseg
import re
jieba.enable_parallel(80)
fout1 = open('/usr/cner/data/example.dev', 'w', encoding='utf8')
fout2 = open('/usr/cner/data/example.test', 'w', encoding='utf8')
fout3 = open('/usr/cner/data/example.train', 'w', encoding='utf8')
dics = csv.reader(open("/home/zhaokunwang/DataLibrary/DICT/DICT_NOW.csv", 'r', encoding='utf8'))
flag = 0
for row in dics:
    if flag == 0:
        flag = 1
        continue
    if len(row) == 2:
        jieba.add_word(row[0].strip(), tag=row[1].strip())
jieba.del_word('否认')
# ss = "器官,定位到器官及更小位置的大小便正常言语障碍,胃、胸闷、支气管、左心室"
# res = pseg.lcut(ss, HMM=False)
# for k, v in res:
#     print("k,v pair is %s,%s" % (k, v))

files = open('/home/zhaokunwang/DataLibrary/Entity_Tagging/data/source.csv', 'r', encoding='utf8')
csvfile = csv.reader(files)
fuhao = ['；', '。', '?', '？', '!', '！', ';']
biaoji = ['DIS', 'SYM', 'SGN', 'TES', 'DRU', 'SUR', 'PRE', 'PT', 'Dur', 'TP', 'REG', 'ORG', 'AT', 'PSB', 'DEG', 'FW',
          'CL']
dd = 0
count=0
split_num = 0
for row in csvfile:
    count=count+1
    if count>31:
        break;
    if dd == 0:
        dd = 1
        continue
    split_num += 1
    if split_num % 15 == 0:
        index = str(1)
    elif split_num % 15 > 0 and split_num % 15 < 4:
        index = str(2)
    else:
        index = str(3)

    content = row[4]
    line_num = []
    list_content = re.split('\n', content.strip())
    if list_content:
        for oneline in list_content:

            aline = re.sub(' |\t|━|_|─|□|─|─|_| ', '', oneline.strip())
            if len(aline) > 5:
                line_seg = pseg.lcut(aline, HMM=False)

                for key, value in line_seg:
                    if value.strip() and key.strip():

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
                            begin = 0
                            for char in key.strip():
                                if begin == 0:
                                    begin += 1
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