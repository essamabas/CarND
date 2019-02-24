# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
[output1]: ./examples/output_9_0.png "output1"
[output2]: ./examples/output_10_0.png "output2"
[output3]: ./examples/output_16_1.png "output3"
[output4]: ./examples/output_31_2.png "output4"
[output5]: ./examples/output_34_0.png "output5"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 4410
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43


#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. The figure shows a sample of the available Image-Labels.
![alt text][output1]

To view It is a bar chart showing how the data for each Label is distributed.
![alt text][output2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)


1. As an initial model architecture the original LeNet model from the course was chosen. In order to tailor the architecture for the traffic sign classifier usecase I adapted the input so that it accepts the colow images from the training set with shape (32,32,3) and I modified the number of outputs so that it fits to the 43 unique labels in the training set. The training accuracy was 83.5% and my test traffic sign “pedestrians” was not correctly classified. *(used hyper parameters: EPOCHS=10, BATCH_SIZE=128, learning_rate = 0,001, mu = 0, sigma = 0.1)*

2. After adding the grayscaling preprocessing the **validation accuracy increased to 91%** (hyperparameter unmodified)
Here is an example of a traffic sign image after grayscaling.

![alt text][output3]

3. The additional normalization of the training and validation data resulted in a minor increase of **validation accuracy: 91.8% **(hyperparameter unmodified)I normalized the image data 
using `(pixel - 128)/ 128` formula in order to increase the validation accuracy.


#### Data Generation

To add more data to the the data set, I used the following techniques:
1. Images has been randomaly rotated with `30-degree` range.
2. Images has been randomaly translated with `10-pixels`.
3. New generated Images has been appended to the original Test Dataset.
4. Shuffle data to improve the trainning performance.



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The model architecture is based on the LeNet model architecture. 
I added dropout layers before each fully connected layer in order to prevent overfitting. 

<figure>
 <img src="LeNet5_CNN.png" width="880" alt="Combined Image" />
 <figcaption>
 <p></p> 
 <p style="text-align: center;"> Your output should look something like this (above)</p> 
 </figcaption>
</figure>
 <p></p> 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten	      	| outputs 400 				|
| Fully connected		| outputs 120        									|
| RELU					|												|
| Dropout					|												|
| Fully connected		| outputs 84        									|
| RELU					|												|
| Dropout					|												|
| Fully connected		| outputs 43        									|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation
sets imply underfitting. A high accuracy on the training set but low accuracy on the validation set implies overfitting.

I used an iterative approach for the optimization of validation accuracy:

1. As an initial model architecture the original LeNet model from the course was chosen. In order to tailor the architecture for the traffic sign classifier usecase I adapted the input so that it accepts the colow images from the training set with shape (32,32,3) and I modified the number of outputs so that it fits to the 43 unique labels in the training set. The training accuracy was 83.5% and my test traffic sign “pedestrians” was not correctly classified. *(used hyper parameters: EPOCHS=10, BATCH_SIZE=128, learning_rate = 0,001, mu = 0, sigma = 0.1)*

2. After adding the grayscaling preprocessing the **validation accuracy increased to 91%** (hyperparameter unmodified)
The additional normalization of the training and validation data resulted in a minor increase of **validation accuracy: 91.8% **(hyperparameter unmodified)

3. Adding dropout after the two fully connected layer has improved the Validation accuracy to 93%

4. I decided to reduce the learning rate and increase of epochs. **validation accuracy = 96,1%** (EPOCHS = 150, BATCH_SIZE = 128, rate = 0,0006, mu = 0, sigma = 0.1)

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.99
* validation set accuracy of 0.963 
* test set accuracy of 0.938

 ![alt text][output4]

I used an iterative approach for the optimization of validation accuracy:

1. As an initial model architecture the original LeNet model from the course was chosen. In order to tailor the architecture for the traffic sign classifier usecase I adapted the input so that it accepts the colow images from the training set with shape (32,32,3) and I modified the number of outputs so that it fits to the 43 unique labels in the training set. The training accuracy was 83.5% and my test traffic sign “pedestrians” was not correctly classified. *(used hyper parameters: EPOCHS=10, BATCH_SIZE=128, learning_rate = 0,001, mu = 0, sigma = 0.1)*

2. After adding the grayscaling preprocessing the **validation accuracy increased to 91%** (hyperparameter unmodified)
The additional normalization of the training and validation data resulted in a minor increase of **validation accuracy: 91.8% **(hyperparameter unmodified)

3. Adding dropout after the two fully connected layer has improved the Validation accuracy to 93%

4. I decided to reduce the learning rate and increase of epochs. **validation accuracy = 96,1%** (EPOCHS = 150, BATCH_SIZE = 128, rate = 0,0006, mu = 0, sigma = 0.1)



### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

 ![alt text][output5]

The fourth image might be difficult to classify because it includes german text that was not availabe in the trainning dataset.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 14: Stop Sign      		| 14: Stop sign   									| 
| 33: Turn right ahead     			| 33: Turn right ahead 										|
| 4: Speed limit (70km/h)					| 4: Speed limit (70km/h)											|
| 18: General caution	      		| **8: Speed limit (120km/h)**					 				|
| 25: Road work			| 25: Road work      							|
| 28: Children crossing			| 28: Children crossing      							|
| 24: Road narrows on the right	      		| **29: Bicycles crossing**					 				|


The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83%. This compares favorably to the accuracy on the test set of 93%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the corrected classified images, the model was relatively sure of the classification (probability bigger than 0.93), only the 4th image the model gives a probability of highest probability of (0.56 to "Speed limit (120km/h)") - which indicates that the model did not perform well on distorted images - or even images with additional texts.

predictions= [[9.39819813e-01 5.98139130e-02 1.89692262e-04 1.42076591e-04
  3.45599910e-05]
 [9.93675411e-01 5.19704865e-03 7.48918275e-04 2.31663231e-04
  5.17812987e-05]
 [1.00000000e+00 1.94419161e-19 1.08876209e-22 1.56284764e-24
  3.58295704e-29]
 [5.62137127e-01 2.47013330e-01 7.93333054e-02 4.97727357e-02
  3.50560546e-02]
 [9.99995351e-01 4.27816212e-06 3.82426094e-07 6.82602249e-13
  4.52173618e-13]
 [9.94937181e-01 5.03949495e-03 2.32161983e-05 1.07298256e-07
  5.85730531e-09]
 [9.98225987e-01 1.63969817e-03 7.58151073e-05 5.19430832e-05
  4.66089023e-06]]

  prediction_Indicies= [[14 33  3 13 39]
 [33 35 11 14  3]
 [ 4  8  1 15 14]
 [ 8  4  2  1  0]
 [25 29 22 18 20]
 [28 29 27 31 18]
 [29 31 19 28 18]]


