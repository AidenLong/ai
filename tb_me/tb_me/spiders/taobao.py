# -*- coding: utf-8 -*-
import scrapy
import json
import re

from tb_me.items import TaobaoGoodItem


class TaobaoSpider(scrapy.Spider):
    name = 'taobao'
    allowed_domains = ["taobao.com", "alicdn.com", 'tce.taobao.com']
    start_urls = ['https://www.taobao.com/']
    headers = {
        ':authority': 's.taobao.com',
        ':method': 'GET',
        ':path': '/list?q=%E6%97%97%E8%A2%8D&cat=16&style=grid&seller_type=taobao&spm=a219r.lm874.1000187.1',
        ':scheme': 'https',
        'accept': '*/*',
        'accept-encoding': 'gzip,deflate,sdch,br',
        'accept-language': 'zh-CN,zh;q=0.8',
        'cache-control': 'max-age=0',
        'cookie': 'miid=1078783718138049660; tg=0; cna=B4vWELZrz0sCAXEtXX6pb0J7; thw=cn; enc=26FDQ6APQHNF5j5XIU%2FTMV6jfFbLU%2BzrRfn0H1usfY4Ts2KTOL5xhvueJ3wCB4%2FD%2B4dod0syAEYu%2BBxB6ZCTpw%3D%3D; hng=CN%7Czh-CN%7CCNY%7C156; t=cae3d666d300fd33eee3e9affd7dc737; _uab_collina=155471709854810911962161; UM_distinctid=169fc5f287d81-0d70671052dd67-454c092b-100200-169fc5f287ec5; _cc_=W5iHLLyFfA%3D%3D; _m_h5_tk=3c5cb5bcf0beb0efbfc1483314ebac85_1554784094341; _m_h5_tk_enc=6c272369b22a0de0b324e808f5a3ac7d; mt=ci=0_0; cookie2=1b371f9e5fb325593743be1619f68e08; v=0; _tb_token_=e71439e933bfa; x5sec=7b227365617263686170703b32223a223837323137623435353462626638383261626466376138306234323061386561434f477073655546455075617971326f2f642f5243526f4d4d6a59794f4459314e5451304e7a7378227d; JSESSIONID=E5501D7073ACE41A9A068477528BBC74; isg=BHd3Ggtc5-JBpmctMAaIZpCKBmsBlEv6ir9xEckkk8ateJe60Qzb7jVaWpiDiyMW; l=bBjut0DRvn4b8u72BOCanurza77OSIRYYuPzaNbMi_5Q96T62RbOlMLTjF96Vj5RsYYB4KzBhYp9-etkZ',
        'referer': 'https://s.taobao.com/list/_____tmd_____/verify/?nc_token=cdfd0c5931cdd21bd9cb3c70fbeb0665&nc_session_id=01oW7uhg4xsiIun2wMOEE5nbDcZW3e9lFykwAJ1dXtz4wKC0GOIDqiT4QIG65Bq1ViN8HPWMx6QlMWV-Nw9LKhFln6-0AXmw2z9nEi93BnEETuOpmU-HKXxAnvIKFhOJMlr_BiyFgli8QQLds_HXXVAMUyLsahZ2Xl1kZ25HDpM1mZqOZJoQXH0n8hxjsUcU-yHvJG0DpdSHZOdyTDPcKc6A&nc_sig=0529P2i24A0W_fVoh6Q3pCGc45zdB98cnYBnx8Ddfsy3Y-eP0GkR8g-V0uFewn7a2jWmR8QfMSfs9OvKQOSNhxqh34VvCU5rWm5Yiay0hcCw4WgR7LQq2b6WwKT8WHIxb3lmEzqEUq4wf335BJTFLDQ4LNdqBA_YOOBNFvHupb_W_Im_YH7OUSij96SHEwd9HKhzIiGJzKxLpS1LnjuS6F8jnEPs_3D0-lJiwaplLuVDZPVCwK-KfSIkwojhflq9JhozfPbdvl6XRwJT4-VmoHNfzWBFTiYc_AGpvv4wwyzpBGgV1SznEfXFwgTqwe0Kc6zj2_jGsx8iAnzHIgU1Omch9GO3rAlr97uyLxtketbd0o2o3GOxbowBqfBsLfhdzrP-2Wc1ScKXqfnzppw1d251sAxQHojeOF39MSL4bxBD4FHUTk6lcMbGhqrx8iEU-0X5Xwloc1CTqah16zaOjuB5kCZPwJ7fpYHDMUhjy4jQQ&x5secdata=5e0c8e1365474455070961b803bd560607b52cabf5960afff39b64ce58073f780315a4e0d1d443e6302302f1f5698ee439630842aedb51a8ec4eda2e52a1c6a0f4771648dd317841d6c4a57ae30ba7f6e56639bfdbc7c0467001cdb54971729081de66ecf56ceefe58b43bc7d00cbc44524b2377b38c5a19331b3954a03b72e6ce83347b4ada99898b77918eccb9915e691bae1213c60c113af7da7d4a97c75bfeac98bf7624ba1100b1e9195521f4a65fd102e32437f7b8741ec5aa54e4ecf0ffe87f31c552fe362e2ba98dac74a29fd752e9fbd22ab5197b187e7636309ab901e4ee5c156d344e73c99d13eb28e7f0834483b45f7c195c851e5ea867461b6ef58a49c29fa722359df311fdf60c67e643057e2edd1838e1ccc05f4f01cfb617a757579fe65d222fa76fd2040ff84986d0a9b23f28a9cd3eab8be9de602578544c064a42424fa07de3912d37fe56189e1ef6064a3134649e8032b2082dd627fcfb864cf96054ef0c250d0391c4fb7eda43381952952e44c4a23690d7ca227233b383147fc6f55ea311c83f4384fac3caf362578828841d206b4bb143add24363ed32ac1ae80b95823ebd5541d26f474561654ee929b84b62d02675e9344cabeac079de1913fcc2c10c58253cba081d31b24749603dfa93a95d2702cbc2acfc321485c4e540eeecf0b783c921696f3626fa16129ce3f687c1f4410efd5a07011ef28a537c5c0939dfa4629c0a7b58e8df&x5step=100&nc_app_key=X82Y__b6255a2d60da6c0b0b0eab625a47d5ca',
        'upgrade-insecure-requests': 1,
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.87 Safari/537.36'
    }

    # 1.分类页面获取
    def parse(self, response):
        # (1)实例化选择器Selector
        selector = scrapy.Selector(response)
        # (2)获取每一个大类的a标签中的data-dataid值
        dataid_list = []
        dataid_list = selector.xpath('//ul[@class="service-bd"]/li/a/@data-dataid').extract()
        # （3）循环dataid_list，拼接小类的地址,获取数据
        # for dataid in dataid_list:
        # 只获取一个分类的信息
        dataid = dataid_list[0]
        url = 'https://tce.taobao.com/api/data.htm?ids=' + dataid
        # 请求链接，将获取的结果交给parse_cat方法去处理
        # dont_filter取消验证，meta传递参数
        yield scrapy.Request(url=url, callback=self.parse_cat, dont_filter=True, meta={'dataid': dataid})

    # 2.处理分类返回的数据
    def parse_cat(self, response):
        # （1）获取页面内容
        body = response.body
        # （2）JSON解析,引入json模块
        cat_list = json.loads(body)
        # （3）获取value下的list
        dataid = response.meta['dataid']
        list = cat_list[dataid]['value']['list']
        # (4)遍历列表得到子分类列表页面的链接，并且提交给parse_cat_list方法进行进一步处理
        for value in list:
            # 获取link的值，使用get方法，如果没有获取到，那么我们就设置为空
            href = value.get('link', '')
            # 如果非空
            if href:
                # 判断https:子串是否在链接里
                if 'https:' in href:
                    yield scrapy.Request(url=href, callback=self.parse_cat_list, dont_filter=True,
                                         meta={'now_url': href},
                                         headers=TaobaoSpider.headers)
                else:
                    # 不过链接中不存在https:，那么我们就添加https:到链接中
                    yield scrapy.Request(url="https:" + href, callback=self.parse_cat_list, dont_filter=True,
                                         meta={'now_url': "https:" + href},
                                         headers=TaobaoSpider.headers)
            else:
                # 如果没有得到Link的值
                print('-' * 100, '如果没有得到Link的值')

    # 3.子分类列表中获取商品列表
    def parse_cat_list(self, reponse):
        # (1) 获取网页源代码
        text = reponse.text  # extract()返回的是一个列表
        # （2）通过正则表达式，获取所需的那个json串
        if text:
            # 提取g_page_config =  及 };  中间的内容
            cop = re.compile('g_page_config =(.*?)};')
            json_text = re.findall(cop, text)  # 返回的是一个列表
            # 判断正则表达式是否得到了结果
            if json_text:
                # 获取得到的结果，并且还原最后的 }
                json_dict = json.loads(json_text[0] + '}')
                mods = json_dict.get('mods', [])
                # print(mods)

                # ****可以使用连续get方法缩减代码行数
                # auctions=json_dict.get('mods',[]).get('itemlist',[]).get('data',[]).get('auctions',[])
                # #循环获取数据
                # for item in auctions:
                #     datail_href=item.get('detail_url','')
                #     if datail_href:
                #         #如果https在链接中，则直接请求，否则拼接字符串
                #         if "https:" in datail_href:
                #             yield scrapy.Request(url=datail_href,callback=self.parse_good,dont_filter=True)
                #         else:
                #             yield scrapy.Request(url="https:"+datail_href, callback=self.parse_good,
                #                                  dont_filter=True)
                # 判断mods是否存在
                if mods:
                    itemlist = mods.get('itemlist', [])
                    if itemlist:
                        data = itemlist.get('data', [])
                        if data:
                            auctions = data.get('auctions', [])
                            # 循环获取数据,获取商品的列表
                            # for item in auctions:
                                # datail_href = item.get('detail_url', '')
                            datail_href = auctions[0].get('detail_url', '')
                            if datail_href:
                                # 如果https在链接中，则直接请求，否则拼接字符串
                                print(datail_href)
                                if "https:" in datail_href:
                                    yield scrapy.Request(url=datail_href, callback=self.parse_good,
                                                         dont_filter=True)
                                else:
                                    yield scrapy.Request(url="https:" + datail_href, callback=self.parse_good,
                                                         dont_filter=True)
                    # 分页处理
                    # pager_data = mods.get('pager', []).get('data', [])
                    # if pager_data:
                    #     pageSize = pager_data.get('pageSize', 60)  # 每一页显示的数量
                    #     totalPage = pager_data.get('totalPage', 30)  # 总页数
                    #     currentPage = pager_data.get('currentPage', 1)  # 当前页数
                    #     next_href = reponse.meta['now_url'] + "&s=" + str(int(currentPage) * int(pageSize))
                    #     # 请求自身，得到更多其他页的数据
                    #     # 拼接下一页的链接
                    #     # 当前页数小于下一页的页数
                    #     if currentPage < totalPage:
                    #         # 如果进入下一页的处理
                    #         print('/' * 100 + '\n' + next_href)
                    #
                    #         yield scrapy.Request(url=next_href, callback=self.parse_cat_list, dont_filter=True,
                    #                              meta={'now_url': reponse.meta['now_url']})
            else:
                # 没有得到json
                print('*' * 100, '没有找到g_page_config')
        else:
            # 如果得不到商品)的json串
            print('+' * 100, '子分类列表没有找到')

    # 3.商品信息获取
    def parse_good(self, response):
        # （1）大小分类名称的获取（自行完成）
        # （2）商品名称
        # div[ @ id = 'J_Title'] / h3 / text()
        # （3）商品备注
        # div[ @ id = 'J_Title'] / p / text()
        # （4）价格
        # a.价格
        # *[ @ id = 'J_StrPrice'] / em[@class ='J_StrPrice'] / text()
        # b.淘宝价
        # （暂时放弃）
        # （5）发货地址
        # （暂时放弃）
        # （6）累计评论
        # （7）交易成功量
        # （8）宝贝详情
        # ul[@ class ='attributes-list'] / li / text()
        sel = scrapy.Selector(response)
        # 商品名称
        title = sel.xpath("//div[@id='J_Title']/h3/@data-title").extract()
        if title:
            good_title = title[0]
        else:
            good_title = ""
        # 商品备注
        commout = sel.xpath("//*[@id='J_Title']/p/text()").extract()
        if commout:
            good_commout = commout[0]
        else:
            good_commout = ''
        # 商品价格
        price = sel.xpath("//*[@id='J_StrPrice']/em[@class='J_StrPrice']/text()").extract()
        if price:
            good_price = price[0]
        else:
            good_price = 0.0
        # 商品详情
        info = sel.xpath("//ul[@class='attributes-list']/li/text()").extract()
        good_info = "\t".join(info)

        # 获取图片URL
        image_list = sel.xpath('//ul[@id="J_UlThumb"]/li/div/a/img/@data-src').extract()
        if not image_list:
            # 为空时
            image_list = []

        # 动态请求数据详细页面，使用自定义的header包，传递已抓取的数据，实现数据绑定
        # （1）讲已抓取的数据打包
        good_dict = {}
        good_dict['title'] = good_title
        good_dict['commout'] = good_commout
        good_dict['price'] = good_price
        good_dict['info'] = good_info
        good_dict['href'] = response.url
        good_dict['image_url'] = image_list

        # (2)获取itemid和sellerid
        itemId_list = sel.xpath('//*[@id="J_Pine"]/@data-itemid').extract()
        if itemId_list:
            itemId = itemId_list[0]
            good_dict['itemId'] = itemId
            sellerId = sel.xpath('//*[@id="J_Pine"]/@data-sellerid').extract()[0]
            # (3)建议自定义的header
            header = {
                ':authority': 'detailskip.taobao.com',
                ':method': 'GET',
                ':path': '/service/getData/1/p1/item/detail/sib.htm?itemId={0}&sellerId={1}&modules=dynStock,qrcode,viewer,price,duty,xmpPromotion,delivery,upp,activity,fqg,zjys,couponActivity,soldQuantity,originalPrice,tradeContract&callback=onSibRequestSuccess'.format(
                    itemId, sellerId),
                ':scheme': 'https',
                'accept': '*/*',
                'accept-encoding': 'gzip,deflate,sdch,br',
                'accept-language': 'zh-CN,zh;q=0.8',
                'cache-control': 'max-age=0',
                'cookie': 'miid=1078783718138049660; tg=0; ubn=p; ucn=center; cna=B4vWELZrz0sCAXEtXX6pb0J7; thw=cn; enc=26FDQ6APQHNF5j5XIU%2FTMV6jfFbLU%2BzrRfn0H1usfY4Ts2KTOL5xhvueJ3wCB4%2FD%2B4dod0syAEYu%2BBxB6ZCTpw%3D%3D; hng=CN%7Czh-CN%7CCNY%7C156; t=cae3d666d300fd33eee3e9affd7dc737; UM_distinctid=169fc5f287d81-0d70671052dd67-454c092b-100200-169fc5f287ec5; _cc_=W5iHLLyFfA%3D%3D; mt=ci=0_0; cookie2=1b371f9e5fb325593743be1619f68e08; v=0; _tb_token_=e71439e933bfa; _m_h5_tk=3c75e9b7e53e6b4742fd0f806c438ddc_1554806740784; _m_h5_tk_enc=cdc06f6429495d8339dd2be8b25460e4; x5sec=7b2264657461696c736b69703b32223a226538633064623164363863363736363466366438356334386264366364326662434a537373655546454d627371632f677372576d30774561444449324d6a67324e5455304e4463374d513d3d227d; isg=BNLSjtmj2sFuTyJ-Rckl5a09I5h0S9YZp0h0opwregVwr3GphHM3jT2OH0s2304V; l=bBjut0DRvn4b8cV1BOfZSuI8aob9oIRb8sPzw4_GqICP_bfw52b5WZsp7TTeC3GVZ6s6R3J4RKM8BdZvqyC5.',
                'referer': 'https://item.taobao.com/item.htm?spm=a219r.lm874.14.10.30361f90UIFkv4&id=573066310472&ns=1&abbucket=20',
                'user-agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36'
            }
            # 3.请求拼接
            href = 'https://detailskip.taobao.com/service/getData/1/p1/item/detail/sib.htm?itemId={0}&sellerId={1}&modules=dynStock,qrcode,viewer,price,duty,xmpPromotion,delivery,upp,activity,fqg,zjys,couponActivity,soldQuantity,contract,tradeContract'.format(
                itemId, sellerId)
            # 4.页面请求
            yield scrapy.Request(url=href, headers=header, meta={'item': good_dict}, callback=self.parse_good_info,
                                 dont_filter=True)
        else:
            # 代表没有获取到itemId
            print('+/' * 100, '没有获取到itemId')

    def parse_good_info(self, reponse):
        # 显示页面的主体内容
        # print(reponse.body)
        # print('*'*100)
        # print(reponse.url)
        # 获取发货城市
        json_dict = json.loads(reponse.body)
        sendCity = json_dict['data']['deliveryFee']['data']['sendCity']
        # 交易成功量
        confirmGoodsCount = json_dict['data']['soldQuantity']['confirmGoodsCount']
        # 价格
        price = json_dict['data']['price']
        # 淘宝价格
        promoData = json_dict['data']['promotion']['promoData']
        if promoData:
            tb_price_def = promoData['def']
            if tb_price_def:
                tb_price = tb_price_def[0]['price']
            else:
                tb_price = 0
        else:
            tb_price = 0
        # 5.存储数据
        # 在items中定义一个item用来和管道进行通信
        # 实例化自定义的Item
        item = TaobaoGoodItem()
        item['title'] = reponse.meta['item']['title']
        item['commout'] = reponse.meta['item']['commout']
        if reponse.meta['item']['price'] == 0.0:
            item['price'] = price
        else:
            item['price'] = reponse.meta['item']['price']
        item['sendCity'] = sendCity
        item['confirmGoodsCount'] = confirmGoodsCount
        item['info'] = reponse.meta['item']['info']
        item['itemId'] = reponse.meta['item']['itemId']
        item['href'] = reponse.meta['item']['href']
        item['tb_price'] = tb_price
        item['image_urls'] = reponse.meta['item']['image_url']
        href = 'https://rate.taobao.com/detailCount.do?itemId={0}'.format(reponse.meta['item']['itemId'])
        yield scrapy.Request(url=href, meta={'item': item}, dont_filter=True, callback=self.parse_good_detail)

    def parse_good_detail(self, response):
        # json 解析,respone.body传递来的是得到页面，恰好这个页面是一个json的字符串
        # 去除前面的null(  以及最后的 )
        body = response.body
        json_dict = json.loads(body[5:-1])
        # 得到上一级传递过来的item
        item = response.meta['item']
        item['detailCount'] = json_dict['count']
        yield item
    # 4.存储数据
    # 5.下载商品图片
