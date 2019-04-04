import urllib.request
def use_proxy(proxy_addr,url):
    #1.ProxyHandler()设置对应的代理服务器信息
    proxy=urllib.request.ProxyHandler({'http':proxy_addr})
    #2.build_opener()创建一个自定义的opener对象
    opener=urllib.request.build_opener(proxy)
    #3.将opener加载为全局使用的opener对象
    urllib.request.install_opener(opener)
    #4.发送请求
    data=urllib.request.urlopen(url).read().decode('utf-8')
    return data
proxy_addr='61.135.217.7:80'
url='http://www.taobao.com'
print(len(use_proxy(proxy_addr,url)))
print(use_proxy(proxy_addr,url))