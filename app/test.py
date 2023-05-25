import requests

url = 'http://127.0.0.1:5000/clasification'
files = {'text': open('app/texto.txt', 'rb')}

response = requests.post(url, files=files)

print(response.text)
