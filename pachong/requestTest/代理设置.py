import requests
proxies={
    "http":"61.135.217.7:80"
}
req=requests.get('http://www.taobao.com/',proxies=proxies)
print(req.text)
