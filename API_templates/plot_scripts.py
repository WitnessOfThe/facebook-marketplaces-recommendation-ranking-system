from IPython.display import Image as Im
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

def plot_image_with_category(df,file_path,file_name,category_str):
    fig = plt.figure(figsize=(5., 5.))
    grid = ImageGrid(fig, 111, 
                    nrows_ncols=(1, 1),  # creates 2x2 grid of axes
                    axes_pad=0.1,  # pad between axes
                    )
    img_arr = []
    img_arr.append(Image.open(file_path + file_name + '.jpg'))

    for ax, im in zip(grid, img_arr):
        ax.imshow(im)
        ax.text(0, 30, 'Predicted cat:'+category_str, style='italic',
            bbox={'facecolor': 'red', 'alpha': 0.75, 'pad': 10})
        ax.text(0, 150, 'Real cat:'+get_real_category(df,file_name)[0], style='italic',
            bbox={'facecolor': 'green', 'alpha': 0.75, 'pad': 10})
    plt.show()

def get_real_category(df,file_name):
    return df[df['id_x'] == file_name]['cat:0'].values.tolist()

def plot_images_with_category(df,resp_dic,file_path,cat_list):
    fig = plt.figure(figsize=(10., 10.))
    grid = ImageGrid(fig, 111, 
                    nrows_ncols=(2, 2),  # creates 2x2 grid of axes
                    axes_pad=0.1,  # pad between axes
                    )
    
    img_arr = []
    for im in resp_dic['image_labels']:
        img_arr.append(Image.open(file_path + im + '.jpg'))
    i =0 
    for ax, im in zip(grid, img_arr):
        ax.imshow(im)
        ax.text(0, 30, 'Predicted cat:'+cat_list[i], style='italic',
            bbox={'facecolor': 'red', 'alpha': 0.75, 'pad': 10})
        ax.text(0, 150, 'Real cat:'+get_real_category(df,resp_dic['image_labels'][i])[0], style='italic',
            bbox={'facecolor': 'green', 'alpha': 0.75, 'pad': 10})
        i +=1
    plt.show()