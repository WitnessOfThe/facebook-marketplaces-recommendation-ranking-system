# Facebook Marketplace Image based Recommendation System

In this progect we impement fastAPI based API in Docker container deployed in Amazon Cloud. This API provides methods that allow to categorise images into 13 product categories and search for the similar images through the image database. The categorisation model is based on the [ResNet50](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html) neural network, while indexing is performed by FAISSE indexing system. 

API Methods

<details>
    <summary>GET status</summary>
    https://github.com/WitnessOfThe/facebook-marketplaces-recommendation-ranking-system/blob/d4a6b261455974b2ef559cc48538ed1ede9a3734/requests_templates/api_health_check.py#L2-L8
</details>
<details>
  <summary>POST image embeding</summary>
</details>
<details>
  <summary>POST image category</summary>
</details>
<details>
  <summary>POST image similar images lavels from base</summary>
</details>

## get similar index

# 1 Data preparation

We have two ".csv" tables and one archive with the graphical data. The text data "Products.csv" contains information about listings on the market grouped by the listing id ("product_id") with their categorisation and description, while the "Images.csv" maps the listing id with image_id stored separately in the archive as ".jpeg".

The cleaning of the ".csv" data starts with the conversion of the "price" column into a proper "float" and removing all raws consisting of missing or NaN data from the dataset "Products.csv". 
The field "category" contains the hierarchical structure separated by " / ", we parse it to extract the root category. 
The root category has to be transformed into the number, so dictionaries "encoder" and "decoder" is implemented.
Finally, the categories is mapped to the image id from "Images.csv", through the "product_id" forming dataset "training_data.csv".

The graphical data has to be homegeneus to be processed by ML algorithm, therefore initial images has to be checked to be the same "RGB" type and the same size. As we realise that all out figures has different resolution and aspect ratio, we need to fix it by resizing each image to the simular size. The class ImagesHandle is responsible for performing this task.

The initial dataset of 11121 categorised images is split into the training (10k images) and test (1121 images) datasets. We split the training data into the evaluation (30%) and training (70%) parts during model training. Each dataset split was performed randomly, so each category is well represented in test and training data. The prepared images are all homogenous, normalised and set to be a size of 256x256. The data augmentation also used adding random image rotations, and vertical and horizontal splits.

The final model performance is ___

# 2 Transfer learning for computer vision

In this project, we use the neural network resnet50, which is available as one of the standard PyTorch models. The resnet50 is a convolutional neural network, which is highly efficient for image classification problems. To increase the efficiency of product categorization, we use a transfer learning approach, where one can take an initially pre-trained neural network and finetune it for a specific problem. In our case, we load weights of the resnet50 model, which is trained on the imagenet dataset to categorize images into the 1k different types. As our have only 13 categories, we need to change the dimensions of the last linear layer and one before to finetune the model to our categariation space.

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
