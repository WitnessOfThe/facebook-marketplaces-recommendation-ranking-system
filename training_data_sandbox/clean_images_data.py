# %%
import pandas as pd
import numpy as np
import cv2
import os
from PIL import Image
import os

class ImagesHandle:

    def __init__(self,path_dirty,path_clean,size=256) -> None:
        self.size       = size
        self.path_dirty = path_dirty
        self.path_clean = path_clean
        self.names      = self.get_list_of_images_names(path_dirty)
        self.start_resize()
        pass

    def start_resize(self):
        i = 0
        for _ in self.names:
            print(i)
            i += 1
            new_im = self.resize_image(self.size, Image.open(self.path_dirty+_))
            new_im.save(self.path_clean+_)

    def resize_image(self,final_size, im):
        size = im.size
        ratio = float(final_size) / max(size)
        new_image_size = tuple([int(x*ratio) for x in size])
        im = im.resize(new_image_size, Image.ANTIALIAS)
        new_im = Image.new("RGB", (final_size, final_size))
        new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
        return new_im


    def get_list_of_images_names(self,path):
        self.names = os.listdir(path)
#        print(self.names)
        for _ in self.names:
            if not('.jpg' in _):
                self.names.remove(_)
        return self.names

    def get_list_of_images_size(self,path):
        width   = []
        height  = []
        channel = []
        i = 0
        for _ in self.names:
            print(i)
            i += 1
            raw = cv2.imread(path+_).shape
            width.  append(raw[0])
            height. append(raw[1])
            channel.append(raw[2])
        frame_size = pd.DataFrame({'width':width,'height':height,'channel':channel,})
        return frame_size

if __name__ == '__main__':
    path_dirty = 'C:/Users/Danila/Documents/GitHub/facebook-marketplaces-recommendation-ranking-system/images_fb/images/'
    path_clean = 'C:/Users/Danila/Documents/GitHub/facebook-marketplaces-recommendation-ranking-system/images_fb/clean_images_224/'
    ih = ImagesHandle(path_dirty,path_clean)
#    ih.get_list_of_images_names(path_dirty)
 #   df_size = ih.get_list_of_images_size(path_dirty)
    pass
#'''    
    
  #  dirs = os.listdir(path)
 #   final_size = 512
   # for n, item in enumerate(dirs[:5], 1):
    #    im = Image.open('images/' + item)
     #   new_im = resize_image(final_size, im)
      #  new_im.save(f'{n}_resized.jpg')

# %%
