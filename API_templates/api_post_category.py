# %%
import requests
import pandas as pd
import os
from plot_scripts import plot_image_with_category

# load true category data for showdown
df = pd.read_csv('training_data_sandbox\\training_data.csv') 
    
#host = 'http://63.33.191.55:8080' # ec2 instance adress
host = 'http://127.0.0.1:8000'   # local instance adress 

file_path = 'images_fb\\images\\' # test dataset with full resolution
file_name = 'ebdb09e9-de15-4b63-aff0-bae01c9cd068' # example with smartwatch
file_name = '4fb45cbc-2ff6-46e8-9bb5-b0d1f345ff70' # DIY

#file_name ='9ecc1b43-da55-4189-87f7-0308424c68f0' # index sanity check

# API Call and print
url = host+'/predict/category'
file = {'file': open(file_path+file_name+'.jpg', 'rb')} 
category_req = requests.post(url=url,files=file)
print(category_req.json())

# Display result
plot_image_with_category(df,file_path,file_name,category_req.json()['category'])
