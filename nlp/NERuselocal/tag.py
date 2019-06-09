import csv
import re
import pandas as pd
import jieba
import jieba.posseg as pseg
jieba.enable_parallel(2)

fout=open('/usr/cner/data/allEMR.txt','w',encoding='utf8')
dics = csv.reader(open('/home/zhaokunwang/DataLibrary/DICT/DICT_NOW.csv','r',encoding='utf8'))
flag=0
for row in dics:
    if flag==0:
        flag=1
        continue
    if len(row)==2:
        jieba.add_word(row[0].strip(),tag=row[1].strip())

#dic=csv.reader(open('/home/zhaokunwang/DataLibrary/DICT/DICT_NOW.csv','r',encoding='utf8'))
ss="器官,定位到器官及更小位置的大小便正常言语障碍,胃、胸闷、支气管、左心室"
flag=0
ff=1

# for word_tag in dic:
#     if word_tag and flag==0:
#         flag+=1
#         continue
#     if len(word_tag)==2:
#         #jieba.add_word()
#         jieba.add_word(word_tag[0].strip(),tag=word_tag[1])
#         flag+=1
# print("add %d taged word to dic " %(flag))
#res=pseg.cut(ss)
res=pseg.lcut(ss,HMM=False)
for k,v in res:
    print("k,v pair is %s,%s" %(k,v))
csvfile=csv.reader(open('/home/zhaokunwang/DataLibrary/Entity_Tagging/data/source.csv','r',encoding='utf8'))
fuhao = ['；', '。', '?', '？', '!', '！', ';']
biaoji=['DIS','SYM','SGN','TES','DRU','SUR','PRE','PT','Dur','TP','REG','ORG','AT','PSB','DEG','FW','CL']
dd=0
for row in csvfile:

    if dd==0:
        dd=1
        continue
    content=row[4]
    list_content=re.split('\n', content.strip())
    if list_content:
        for oneline in list_content:

            aline=re.sub(' |\t|鈹亅_|鈹�鈻�鈹�鈹�_| ', '', oneline.strip())
            print(aline)
            ff+=1
            if len(aline)>8:
                line_seg=pseg.cut(aline,HMM=False)

                for key,value in line_seg:
                    if value.strip() and key.strip():

                        if value not in biaoji:
                            value='O'
                            for achar in key:
                                if achar and achar in fuhao:
                                    string=achar.strip()+" "+value.strip()+"\n"+"\n"
                                elif achar and achar not in fuhao:
                                    string = achar.strip() + " " + value.strip() + "\n"
                                else:
                                    continue
                                fout.write(string)
                        elif value  in biaoji:
                            begin=0
                            for char in key:
                                if begin==0:
                                    begin+=1
                                    string1=char+' '+'B-'+value.strip()+'\n'
                                    fout.write(string1)
                                else:
                                    string1 = char + ' ' + 'I-' + value.strip() + '\n'
                                    fout.write(string1)
                        else:
                            continue

        if ff>10000:
            break
fout.close()




