# %%
import os
import requests

if 'requests_templates' in os.getcwd():
    os.chdir("..")
    root_path = os.getcwd()

host = 'http://127.0.0.1:8000'   # local instance adress 

file_path = root_path+'\\images_fb\\images\\'      # the image from test dataset
file_name = 'ebdb09e9-de15-4b63-aff0-bae01c9cd068' # file example 

url = host+'/predict/feature_embedding'
file = {'file': open(file_path+file_name+'.jpg', 'rb')} # the image embedding
embedings_req = requests.post(url=url,files=file)
print(embedings_req.json())