# Facebook Marketplace Recommendation Ranking System
#1 data preparation. 

We have two ".csv" tables and one archive with the graphical data. The text data "Products.csv" contains information about listings on the market grouped by the listing id ("product_id") with their categorisation and description, while the "Images.csv" maps the listing id with image_id stored separately in the archive as ".jpeg".

The cleaning of the ".csv" data starts with the conversion of the "price" column into a proper "float" and removing all raws consisting of missing or NaN data from the dataset "Products.csv". 
The field "category" contains the hierarchical structure separated by " / ", we parse it to extract the root category. 
The root category has to be transformed into the number, so dictionaries "encoder" and "decoder" is implemented.
Finally, the categories is mapped to the image id from "Images.csv", through the "product_id" forming dataset "training_data.csv".

The graphical data has to be homegeneus to be processed by ML algorithm, therefore initial images has to be checked to be the same "RGB" type and the same size. As we realise that all out figures has different resolution and aspect ratio, we need to fix it by resizing each image to the simular size. The class ImagesHandle is responsible for performing this task.