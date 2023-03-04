# %%
import requests
import pandas as pd
import os
from plot_scripts import plot_image_with_category

# load true category data for showdown
df = pd.read_csv('training_data_sandbox\\training_data.csv') 
    
host = 'http://63.33.191.55:8080' # ec2 instance adress
#host = 'http://127.0.0.1:8000'   # local instance adress 

file_path = 'images_fb\\images\\' # test dataset with full resolution
file_name = 'ebdb09e9-de15-4b63-aff0-bae01c9cd068' # example with smartwatch

# API Call and print
url = host+'/predict/category'
file = {'file': open(file_path+file_name+'.jpg', 'rb')} 
category_req = requests.post(url=url,files=file)
print(category_req.json())

# Display result
plot_image_with_category(df,file_path,file_name,category_req.json()['category'])
