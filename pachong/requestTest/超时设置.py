import requests
try:
    headers={
        'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36'
    }
    r=requests.get('https://www.oschina.net/translate/tag/python',timeout=0.05,headers=headers)
    print(r.status_code)
except Exception as e:
    print(e)