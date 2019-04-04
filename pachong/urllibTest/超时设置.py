import urllib.request

for i in range(1,50):
    try:
        file=urllib.request.urlopen('http://www.ibeifeng.com', timeout=0.2)
        data=file.read()
        print(len(data))
    except Exception as e:
        print('出现异常--》'+str(e))