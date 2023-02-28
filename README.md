# Facebook Marketplace Recommendation Ranking System
# 1 Data preparation

We have two ".csv" tables and one archive with the graphical data. The text data "Products.csv" contains information about listings on the market grouped by the listing id ("product_id") with their categorisation and description, while the "Images.csv" maps the listing id with image_id stored separately in the archive as ".jpeg".

The cleaning of the ".csv" data starts with the conversion of the "price" column into a proper "float" and removing all raws consisting of missing or NaN data from the dataset "Products.csv". 
The field "category" contains the hierarchical structure separated by " / ", we parse it to extract the root category. 
The root category has to be transformed into the number, so dictionaries "encoder" and "decoder" is implemented.
Finally, the categories is mapped to the image id from "Images.csv", through the "product_id" forming dataset "training_data.csv".

The graphical data has to be homegeneus to be processed by ML algorithm, therefore initial images has to be checked to be the same "RGB" type and the same size. As we realise that all out figures has different resolution and aspect ratio, we need to fix it by resizing each image to the simular size. The class ImagesHandle is responsible for performing this task.

# 2 Transfer learning for computer vision

In this project we use neural network resnet50, which is available as one of the standart PyTorch models. The resnet50 is convolutional neural network, which is highly efficient for the image classification problems. To increase the efficiency of product categoresation, we use transfer learning approach, where one can take initiatally pretranined neural network and finetune it for specific problem. In our case, we load weights of resnet50 model, which is trained to categorise images into the 1k different types. As in our case we have only 13 categories, we need to change the dimensions of the last linear layer, as well as layer before the classifier to increase model performance. 

The initial dataset of 11121 categorised images is splitted into the training (10k images) and test (1121 images) datasets. During model trainining, we split the training data into the evaluation (30%) and training (70%) parts. Each dataset split was performed randomly, so each category is well represented in test and training data. The prepared images all homogenouse and set to be a size of 256x256. The data augmentation also used adding random image rotations, vertical and horisontal splits. 

The final model performance is ___

The resulting model is used to create image embeding i.e. progection of the image onto the vector space with 13 categories. The dictionaries of image embedings used to create a FAISSE index, which allow to efficiently search close vectors in n-dimensinal vector space. 

# 4 Api and Docker Deploy
