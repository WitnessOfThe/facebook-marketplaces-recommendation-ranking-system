import pandas as pd
import numpy as np
import cv2
import os
from PIL import Image
import os
import torch
from torch import optim, nn
from torchvision import models, transforms

model    = models.resnet50( weights=("DEFAULT"))

class ImageToModel:
    
    def __init__(self,im,size=256) -> None:
        self.image = self.resize_image(self,size, im)
        self.transform             = transforms.Compose([transforms.RandomVerticalFlip(),
                                                         transforms.RandomRotation((-180,+180)),
                                                         transforms.RandomHorizontalFlip(),
                                                         transforms.ToTensor(),
                                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.image = self.transform(self.image)

    def resize_image(self,final_size, im):
        size = im.size
        ratio = float(final_size) / max(size)
        new_image_size = tuple([int(x*ratio) for x in size])
        im = im.resize(new_image_size, Image.ANTIALIAS)
        new_im = Image.new("RGB", (final_size, final_size))
        new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
        return new_im

def get_features(img,model):
    # Reshape the input image to match the expected shape of the ResNet50 model
    img = ImageToModel(img).image
    # Disable gradient computation to reduce memory consumption
    with torch.no_grad():
        # Pass the image through the ResNet50 model
        features = model(img)

    # Flatten the output tensor
    features = torch.flatten(features, start_dim=1)

    # Return the extracted features
    return features