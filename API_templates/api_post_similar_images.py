# %%
import requests
import pandas as pd
import os
from plot_scripts import plot_images_with_category

if 'requests_templates' in os.getcwd():
    os.chdir("..")
    root_path = os.getcwd()

# load true category data for showdown
df = pd.read_csv('training_data_sandbox\\training_data.csv') 
    
host = 'http://63.33.191.55:8080' # ec2 instance adress
host = 'http://127.0.0.1:8000'   # local instance adress 
file_path = 'images_fb\\images\\' # the image from test dataset with full resolution

file_name = 'ebdb09e9-de15-4b63-aff0-bae01c9cd068' # example with smartwatch
#file_name = 'fef8b3d3-6f53-4f82-8a6e-a4fde2132c7d'

# Make a call
url = host+'/predict/similar_images'
file = {'file': open(file_path+file_name+'.jpg', 'rb')} # the image embedding
resp = requests.post(url=url,files=file)
resp_dic = dict(resp.json())

# get response
print(resp_dic)

# Agregate to plot
cat_list = []
url = host+'/predict/category'
for im in resp_dic['image_labels']:
    file = {'file': open(file_path+file_name+'.jpg', 'rb')} # the image embedding
    category_req = requests.post(url=url,files=file)
    cat_list.append(category_req.json()['category'])

plot_images_with_category(df,resp_dic,file_path,cat_list)
# %%
