# %%
import pandas as pd
import numpy as np
import cv2
import os
from PIL import Image
import os
import torch
from torch import optim, nn
from torchvision import models
from pytorch_handle import CustomImageDataset


if __name__ == '__main__':
    model    = models.resnet50(  weights='IMAGENET1K_V2')
    model.eval()
    full_data= CustomImageDataset()
    for _ in range(len(full_data)):      
#      print(model(full_data[_][0].unsqueeze(0)))
        img,_ = torch.utils.data.DataLoader(full_data[_], batch_size=1)
        print(model(img))    
# %%
