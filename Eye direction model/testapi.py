import requests
import random
import threading
from threading import Thread
from time import sleep
import json
from ua import *


Count = 0

this = open('example.png', 'rb')



r = requests.put("http://localhost:8001", files={
'media': open('example.png', 'rb')
}
)

print(r)
print(r.text)

if r.status_code==200:
    print(r.json()['label'])