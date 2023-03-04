# %%
import requests
import pandas as pd
import os
from plot_scripts import plot_images_with_category

# load true category data for showdown
df = pd.read_csv('training_data_sandbox\\training_data.csv') 
    
host = 'http://63.33.191.55:8080' # ec2 instance adress
#host = 'http://127.0.0.1:8000'   # local instance adress 

file_path = 'images_fb\\images\\' # test dataset with full resolution
file_name = 'ebdb09e9-de15-4b63-aff0-bae01c9cd068' # example with smartwatch

# API Call and print
url = host+'/predict/similar_images'
file = {'file': open(file_path+file_name+'.jpg', 'rb')} # the image embedding
resp = requests.post(url=url,files=file)
resp_dic = dict(resp.json())
print(resp_dic)

# Agregate to plot
cat_list = []
url = host+'/predict/category'
for im in resp_dic['image_labels']:
    file = {'file': open(file_path+file_name+'.jpg', 'rb')} # the image embedding
    category_req = requests.post(url=url,files=file)
    cat_list.append(category_req.json()['category'])

# Plot images
plot_images_with_category(df,resp_dic,file_path,cat_list)
# %%
