# %%
import requests

host = 'http://63.33.191.55:8080' # ec2 instance adress
#host = 'http://127.0.0.1:8000'   # local instance adress 

# API Call and print
url =  host+'/healthcheck'
resp = requests.get(url) 
print(resp.json())
# %%
