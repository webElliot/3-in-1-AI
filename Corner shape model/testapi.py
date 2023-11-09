import requests
import random
import threading
from threading import Thread
from time import sleep
import json
from ua import *
from color import *

Count = 0

this = open('example.png', 'rb')



r = requests.put("http://localhost:8002", files={
'media': open('example.png', 'rb')
}
)

print(r)
print(r.text)

