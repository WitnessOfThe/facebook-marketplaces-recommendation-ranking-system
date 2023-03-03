# %%
import requests

host = 'http://127.0.0.1:8000'   # local instance adress 

url =  host+'/healthcheck'
resp = requests.get(url) 
print(resp.json())
# %%
