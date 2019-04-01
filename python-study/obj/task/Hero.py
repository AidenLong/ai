# -*- coding:utf-8 -*-
'''
练习：亡者农药小游戏

1、创建三个游戏人物，分别是：
属性:
    名字：name,定位：category,血量：Output技能：Skill
英雄：
    铠，战士，血量：1000 技能：极刃风暴
    王昭君，法师 ，血量：1000 技能：凛冬将至
    阿轲，刺客，血量：1000 技能：瞬华

2、游戏场景，分别：

偷红buff，释放技能偷到红buff消耗血量300
solo战斗，一血，消耗血量500
补血，加血200

'''


class Hero():

    # 定义属性
    def __init__(self, name, category, skill, output=1000, score=0):
        self.name = name
        self.category = category
        self.skill = skill
        self.output = output
        self.score = score
        self.blood = output

    # 战斗场景1，偷红BUFF
    def red_buff(self):
        self.output -= 300
        print('%s%s到对面野区偷红BUFF，消耗血量300' % (self.category, self.name))

    # 战斗场景2solo战斗
    def solo(self, n=1):
        self.output -= 500
        if self.output < 0:
            print('%s%s,送了一个人头，血染王者峡谷' % (self.category, self.name))
        else:
            if self.score == 0:
                self.score += n
                print('%s%s solo战斗拿到一血，消耗血量500' % (self.category, self.name))
            else:
                self.score += n
                print('%s%s solo战斗拿收割%d个人头，消耗血量500' % (self.category, self.name, n))

    # 场景三，加血
    def add_xue(self):
        self.output += 200
        print('%s%s被辅助及时奶了一口，加血200' % (self.category, self.name))

    # 查看英雄相惜信息
    def getInfo(self):
        if self.output <= 0:
            print('%s%s,正在复活,拿到%d个人头' % (self.category, self.name, self.score))

        else:
            print('%s%s超神啦！血量还有%d，拿到%d个人头' % (self.category, self.name, self.output, self.score))


# 实例化对象
kai = hero('铠', '战士', '极刃风暴')

# 操作
kai.red_buff()
kai.getInfo()
kai.solo()
kai.getInfo()
kai.add_xue()
kai.getInfo()
kai.solo()
kai.getInfo()