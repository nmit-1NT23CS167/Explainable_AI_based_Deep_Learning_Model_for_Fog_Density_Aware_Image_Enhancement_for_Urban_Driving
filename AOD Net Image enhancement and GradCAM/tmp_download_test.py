import urllib.request
url = 'https://upload.wikimedia.org/wikipedia/commons/6/6e/Miami_fog.jpg'
urllib.request.urlretrieve(url, 'foggy_test.jpg')
print('saved foggy_test.jpg')
