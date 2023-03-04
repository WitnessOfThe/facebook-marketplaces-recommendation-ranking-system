# Facebook Marketplace Image based Recommendation System

In this progect we impement fastAPI based API in Docker container deployed in Amazon Cloud. This API provides methods that allow to categorise images into 13 product categories and search for the similar images through the image database. The categorisation model is based on the [ResNet50](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html) neural network, while indexing is performed by FAISSE indexing system. 

## API Methods

### GET Status

https://github.com/WitnessOfThe/facebook-marketplaces-recommendation-ranking-system/blob/8425fc2f54bd0a422b376bb9ad932ede8cde7976/API_templates/api_get_health_check.py#L2-L10

```  
$ python API_templates/api_get_health_check.py 
{'message': 'API is up and running!'}
```
### POST Image Embeding

https://github.com/WitnessOfThe/facebook-marketplaces-recommendation-ranking-system/blob/8425fc2f54bd0a422b376bb9ad932ede8cde7976/API_templates/api_post_image_embedings.py#L5-L15  

```  
$python API_templates/api_post_image_embedings.py 
{'features': [-4.232490062713623, 0.03285627439618111, 1.6369472742080688, 0.8700776100158691, -1.2164239883422852, -8.073017120361328, -1.5977203845977783, -1.7229307889938354, 0.9121789336204529, 11.23184871673584, 1.3296902179718018, -3.1180896759033203, 4.341047286987305]}
```

### POST Image Category

https://github.com/WitnessOfThe/facebook-marketplaces-recommendation-ranking-system/blob/8425fc2f54bd0a422b376bb9ad932ede8cde7976/API_templates/api_post_category.py#L10-L20

```  
$python API_templates/api_post_category.py 
{'category_index': 9, 'category': 'Phones, Mobile Phones & Telecoms'}
```

### POST Similar Images From Base

https://github.com/WitnessOfThe/facebook-marketplaces-recommendation-ranking-system/blob/8425fc2f54bd0a422b376bb9ad932ede8cde7976/API_templates/api_post_similar_images.py#L10-L21

```
$python API_templates/api_post_similar_images.py 
{'similar_index': [6788, 7159, 5983, 2210], 'image_labels': ['c26d58d9-91d9-4112-9c35-b50b1bf67ce4', '00ca700f-1055-43a1-b288-0193c7518347', '3ec76c1f-8dbc-429c-a7c9-85749227a06c', '136ab3a8-d0f1-4d8f-9a2e-c393d2dbb286']}
```

# Model 

In this project, we use the neural network resnet50, which is available as one of the standard PyTorch models. The resnet50 is a convolutional neural network, which can be utilised for image classification problems. To increase the efficiency of categorization, we use a transfer learning approach, where one can take an initially pre-trained neural network and finetune it for a specific problem. In our case, we load weights of the resnet50 model 'IMAGENET1K_V2', which is trained on the imagenet dataset to perform classification in the 1k different classed. As our have only 13 product categories, we need to change the dimensions of classification layer and retrain model basing on our database. The images for training is provided by the AICore training program and not available for the public. 

## Data preparation

The raw data conta two ".csv" tables and one archive with "*.jpg" images. First table "Products.csv" contains information about listings on the market grouped by the listing id ("product_id") with their categorisation and description, While the second table "Images.csv" maps the listing id with image_id corresponding to the label of the image stored in the archive.

We start with proccessing of the text data. The cleaning starts with the conversion of the "price" column into a proper "float" and removing all raws consisting of missing or NaN data from the table "Products.csv". The field "category" contains the hierarchical structure separated by " / ". For model training we need to extract 
the root category and assign each unique category an integer number. We create dictionieries "decoder.pkl" and "encoder.pkl" to store maps for direct and reverse tranformations.

The "Image.csv" dataset maps products to the images, where for each product there are two images. These images rerpesent photo of product from the different angle, so they can look very simularly. 

Finally, we join two tables by the key "product_id" forming dataset mapping image label with its category. The described transformations can be found in "sandbox.ipynb"

The resolution used to pretrain resnet50 is 224x224. Therefore we resize our images to be the same size. The processing is performed in the script "clean_images_data.py"

The initial dataset of 11121 categorised images is split into the training (10k images) and test (1121 images) datasets. We split the training data into the evaluation (30%) and training (70%) parts during model training. Each dataset split was performed randomly, so each category is well represented in test and training data. The prepared images are all homogenous, normalised and set to be a size of 256x256. The data augmentation also used adding random image rotations, and vertical and horizontal splits.

The final model performance is ___

# 2 Transfer learning for computer vision


% define epoch

The model training require a measure of the model performance, here we use so called cross entropy losses criterion, which is standart for image classification procedures. Then to provide feedback into the model we use the stochastic gradient descent (SGD) method, which returns updated weights to the model more likely to provide convergence to local minima. One of the key parameters of SGD is the learning rate (lr). In the scope of this progect, we compared two different schedulers to control learning rate. First is the step like changing of the learning rate. We start from the lr = 0.01 and decrease it in 10 times every 40 epochs down to lr = 1E-6 . While the second is the cosine annealing method, which is changing the lr following cosine function from lr_max = 0.01 to  lr_max = 1E-6 with full period of 40 epochs. The corresponding learning rate curves are specified below in the log scale

![plot](https://github.com/WitnessOfThe/facebook-marketplaces-recommendation-ranking-system/blob/main/readme_images/lr_curve.PNG)

The correpsonding training loss rates indicates that cosine annealing method is more likely to escape the local minima and has a potential to find higher performing model weights, while the steplike scheduler prone to stuck in the first found minima with lower propability of escaping

![plot](https://github.com/WitnessOfThe/facebook-marketplaces-recommendation-ranking-system/blob/main/readme_images/train_loss_vs_epoch.PNG)

The comparison of the model performance on the evaluation data set, also plays in favour of the cosine annealing method as it reach higher level of performance using less of the computation time compare to 'staircase'. 

![plot](https://github.com/WitnessOfThe/facebook-marketplaces-recommendation-ranking-system/blob/main/readme_images/eval_vs_epoch.PNG)

Finally, comparison of 'cos' vs 'stepper' gives following accuracy on test dataset

* Accuracy of 'cos' = 58%
* Accuracy of 'stepper' = 56%

To improve the accuracy of our model several procedures can be applied

* Use smaller batch size, this will improve accuracy, but slow down the training proccess

% what can be done to imporove

# 3 Indexing
After model training is over, the neural network can privide image embeding i.e. progection of the image onto the vector space 13 dimensions (categories). Once model aplied on the arbitrary image it returns an array of 13 float numbers, which indecates which category it is more likely to fit in. In order to optimise such process, one can use FAISSE index, which is optimised to search for closest match over the base of n-dimensonal vectors. Here our base conatains 10k image embedings of training dataset. 

# 4 Api and Docker Deploy
In order to make indexing avialable for client, we use fastapi instances. We developed two post methods allowing user to obtain image emdedings and list of the closest images in the dataset. The api is deployed in the Docker container at the EC2 server in Amazon Cloud. 
