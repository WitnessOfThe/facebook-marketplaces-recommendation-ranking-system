# Image based Recommendation System
## Facebook Marketplace 

In this project, we implement fastAPI-based API in a Docker container deployed in Amazon Cloud. This API provides methods that allow categorising images into 13 product categories and searching for similar images through the image database. The categorisation model is based on the [ResNet50](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html) neural network, while indexing is performed by the FAISSE indexing system. 

Key technologies used: Resnet50 neural network (Pytorch implementation), FAISS indexing, FastAPI, Docker 

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
After proccessing:
![plot](https://github.com/WitnessOfThe/facebook-marketplaces-recommendation-ranking-system/blob/main/readme_images/query_to_index.png)

### POST Similar Images From Base

https://github.com/WitnessOfThe/facebook-marketplaces-recommendation-ranking-system/blob/8425fc2f54bd0a422b376bb9ad932ede8cde7976/API_templates/api_post_similar_images.py#L10-L21

```
$python API_templates/api_post_similar_images.py 
{'similar_index': [6788, 7159, 5983, 2210], 'image_labels': ['c26d58d9-91d9-4112-9c35-b50b1bf67ce4', '00ca700f-1055-43a1-b288-0193c7518347', '3ec76c1f-8dbc-429c-a7c9-85749227a06c', '136ab3a8-d0f1-4d8f-9a2e-c393d2dbb286']}
```
After proccessing:
![plot](https://github.com/WitnessOfThe/facebook-marketplaces-recommendation-ranking-system/blob/main/readme_images/index_test_query.png)

# Model 

In this project, we use the resnet50 neural network, which is a convolutional neural network and can be used for image classification tasks. To improve the efficiency of categorization, we use the transfer learning approach, which is to take a pre-trained neural network and tunes it for a specific task. In our case, we load the weights of the resnet50 model "IMAGENET1K_V2" which is trained on the imagenet dataset to perform a classification into 1k different classes. Since we have 13 product categories in total, we need to resize the classification layer and retrain the model based on our database. The training images are provided by the AICore training program.

## Data preparation

The raw data contains two ".csv" tables and one "*.jpg" image archive. The first table "Products.csv" lists the market products grouped by listing ID ("product_id") with their classification and description, and the second table "Images.csv" maps the listing ID to the image_id corresponding to the label of the saved image.

Let's start with processing text data. The cleanup starts by converting the "price" column to the correct "float" and removing all raw data, consisting of missing data or NaN data, from the "Products.csv" table. The "category" field contains a hierarchical structure separated by " / ". To train the model, we need to extract
the root category and give each unique category an integer. We create dictionaries "decoder.pkl" and "encoder.pkl" to store maps for forward and reverse conversion. The "Image.csv" dataset maps products to images, where there are two images for each product. These images are photographs of products from different angles, so they may look very similar. Finally, we join the two tables on the "product_id" key, forming a dataset that displays the image tag with its category. The transformations described can be found in "sandbox.ipynb". Our images dataset contains "*.jpg" files of different resolution and aspect ratios. As the resnet50 is trained by the images of size 224x224, we need to transform it into the right resolution. The processing is performed in the script "clean_images_data.py"


## Model Training
The initial dataset of 11121 categorised images is split into the 'work' (10k images) and 'test' (1121 images) datasets. We split the 'work' data into the 'evaluation' (30%) and 'training' (70%) parts during model training. Each dataset split is performed randomly, so each category is well represented in 'test' and 'training' data. 

During the training procedure model update weights coefficients based on its performance on the training dataset. Each epoch (the round of iterations across 'training' data) we test model performance on the 'evaluation' dataset. As soon as we proceed through the desired number of epochs we test our model on our 'test' dataset, which is our final performance indicator. 

### Dataloader
The dataloader used in the standard training routine (torch.utils.data.DataLoader) is the wrapper around torch.utils.data.Dataset object. Therefore we create a class inheriting torch.utils.data.Dataset, where we implement datahandling with respect to the dataloader spec. We use a data augmentation procedure to increase model resistance to overfitting, so each time image passed to training it experiences random rotation and horizontal/vertical flips. Such transformations applied to the 'training' dataset effectively increase its size, but during 'test' and 'evaluation' procedures it provides extra noise making it hard to analyse results.

### Training procedure

The model training requires a measure of the model performance, here we use the so-called cross-entropy losses criterion, which is standard for image classification procedures. Then to provide feedback on the model we use the stochastic gradient descent (SGD) method, which returns updated weights to the model more likely to provide convergence to local minima. One of the key parameters of SGD is the learning rate (lr). In the scope of this project, we compared two different schedulers to control the learning rate. First is the constant ('flat') lr=0.015 across the full training procedure. And the second is the cosine annealing scheduler, which is changing lr by following the cosine ('cos') function from lr_max = 0.015 to  lr_max = 1E-6 with a full period of 100 epochs. We simulated 300 epochs with batch size 200. The corresponding learning rate curves are specified below


![plot](https://github.com/WitnessOfThe/facebook-marketplaces-recommendation-ranking-system/blob/main/readme_images/lr_curve.PNG)

The corresponding training loss rates suggest that the cosine annealing method is more likely to escape the local minima and has the potential to find higher-performing model weights, while the constant scheduler is prone to stick in the first-found minima with a lower probability of finding better-performing one.  

![plot](https://github.com/WitnessOfThe/facebook-marketplaces-recommendation-ranking-system/blob/main/readme_images/train_loss_vs_epoch.PNG)

The comparison of the model performance on the evaluation data set also shows the potential of the cosine annealing method as it reaches a higher level of performance using less computation time compare to a 'flat' scheduler. 

![plot](https://github.com/WitnessOfThe/facebook-marketplaces-recommendation-ranking-system/blob/main/readme_images/eval_vs_epoch.PNG)

Finally, comparison of performance of 'cos' vs 'flat' on the test dataset gives

* Accuracy of 'cos' = 66.28%
* Accuracy of 'flat' = 66.46%

The tiny marginal difference can indicate that the parameters of the cosine annealing method were far from optimal and further work can be done to improve model performance. Such as
* Use a smaller batch size
* Increase the number of epochs (current simulation time performed at GeForce 1050 TI takes around 9.3 hours)
* find the optimal period and amplitude for the cosine annealing method

## Indexing
After model training is over, the neural network can provide image embedding i.e. projection of the image onto the vector space 13 dimensions (categories). Once applied to the arbitrary image it returns an array of 13 float numbers, which indicates which category it is more likely to fit in. In order to optimise such a process, one can use the FAISSE index, which is the tool used to search for the closest match over the base of n-dimensional vectors. Here, we create a database containing 10k image embeddings of of the training dataset.

Let us check the index performance using API calls. First, we perform a sanity check by passing the training dataset image into the index.

![plot](https://github.com/WitnessOfThe/facebook-marketplaces-recommendation-ranking-system/blob/main/readme_images/index_sanity_check_original_image.png)

Then passing that image to the index returns 4 images, one of which is the original
![plot](https://github.com/WitnessOfThe/facebook-marketplaces-recommendation-ranking-system/blob/main/readme_images/index_sanity_check.png)

Now, we pass user generated image of computer monitor
![plot](https://github.com/WitnessOfThe/facebook-marketplaces-recommendation-ranking-system/blob/main/readme_images/photo_2023-03-05_00-38-50.png)
and get the following response
![plot](https://github.com/WitnessOfThe/facebook-marketplaces-recommendation-ranking-system/blob/main/readme_images/userimage_response.png)
A noticeable feature, that we got three images of a laptop with turned on display
