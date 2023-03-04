# %%
import os
import requests

host = 'http://63.33.191.55:8080' # ec2 instance adress
#host = 'http://127.0.0.1:8000'   # local instance adress 

file_path = 'images_fb\\images\\'                  # test dataset with full resolution
file_name = 'ebdb09e9-de15-4b63-aff0-bae01c9cd068' # example with smartwatch

# API Call and print
url = host+'/predict/feature_embedding'
file = {'file': open(file_path+file_name+'.jpg', 'rb')} # the image embedding
embedings_req = requests.post(url=url,files=file)
print(embedings_req.json())