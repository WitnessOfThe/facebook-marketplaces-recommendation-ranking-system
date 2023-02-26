# %%
import pandas as pd
import numpy as np
import cv2
import os
from PIL import Image
import json 
import os
import torch
import torchvision.transforms as transforms
from torch import optim, nn
from torchvision import models
from pytorch_handle import CustomImageDataset

def get_list_of_images_names(path):
    names = os.listdir(path)
#        print(self.names)
    for _ in names:
        if not('.jpg' in _):
            names.remove(_)
    return names

def call_model(path,name):
    device   = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#
    model    = models.resnet50(  weights='IMAGENET1K_V2')
    model.fc = torch.nn.Linear(model.fc.in_features, 13)
    model.load_state_dict(torch.load( os.path.join(path,name)))

    return model

class ImagePrep:

    def __init__(self,img):
        self.transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.img = self.transform(self.resize_image(256,img)).unsqueeze(0)
        

    def resize_image(self,final_size, im):
        size = im.size
        ratio = float(final_size) / max(size)
        new_image_size = tuple([int(x*ratio) for x in size])
        im = im.resize(new_image_size, Image.ANTIALIAS)
        new_im = Image.new("RGB", (final_size, final_size))
        new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
        return new_im
    
    def transform(self):
        return 
    


if __name__ == '__main__':
    
    model   = call_model('model_final','model_final.pt')

    path    = 'images_fb\\clean_images_256'
    im_list = get_list_of_images_names( 'images_fb\\clean_images_256')
    im_emb_dict = {}
    
    for i,_ in enumerate(im_list):#range(len(im_list)):#
        im_path = os.path.join(path,_)
        im = Image.open(im_path)
        im = ImagePrep(im).img
        emb       = model(im)        
        im_emb_dict[_.replace('.jpg','')] = emb.tolist()[0]
        print(i)

    json_data = json.dumps(im_emb_dict)    
    with open('image_embeddings.json', 'w') as f:
        f.write(json_data)