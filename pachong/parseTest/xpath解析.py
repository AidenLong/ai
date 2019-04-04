from lxml import etree
text='''
    <html>
        <head>
            <title>春晚</title>
        </head>
        <body>
            <h1 name="title">个人简介</h1>
            <div name="desc">
                <p name="name">姓名：<span>小岳岳</span></p>
                <p name="addr">住址：中国 河南</p>
                <p name="info">代表作：五环之歌</p>
            </div>
'''
#初始化
html=etree.HTML(text)
# result=etree.tostring(html) #返回字节流数据
# print(result.decode('utf-8'))

#（1）查询所有的P标签
p_x=html.xpath('//p')
# print(p_x)

#（2）查询所有Name属性的值
v_attr_name=html.xpath('//@name')
# print(v_attr_name)

#(3) 查询所有包含name 属性的标签
e_attr_name=html.xpath('//*[@name]')
# print(e_attr_name)

# (4)查询所有包含name属性，并且name属性值为desc的标签
e_v_attr_name=html.xpath('//*[@name="desc"]')
# print(e_v_attr_name)

#(5)查询所有p标签的文本内容,不包含子标签
p_t=html.xpath('//p')
# for p in p_t:
#     print(p.text)

#(6)查询多个p标签下的所有文本内容，包含子标签中的文本内容
p_m_t=html.xpath('//p')
# for p2 in p_m_t:
#     print(p2.xpath('string(.)'))

#（7）拿到第三个p标签的name属性值
# print(html.xpath('//div/p[3]/@name'))