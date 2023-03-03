# %%
import os
import requests

os.chdir("..")
root_path = os.getcwd()

host = 'http://127.0.0.1:8000'   # local instance adress 

file_path = root_path+'\\images_fb\\clean_images_224\\' # the image from test dataset
file_name = 'ebdb09e9-de15-4b63-aff0-bae01c9cd068'

url =  host+'/healthcheck'
resp = requests.get(url) 
print(resp.json())

url = host+'/predict/feature_embedding'
file = {'file': open(file_path+file_name+'.jpg', 'rb')} # the image embedding
embedings_req = requests.post(url=url,files=file)
print(embedings_req.json())