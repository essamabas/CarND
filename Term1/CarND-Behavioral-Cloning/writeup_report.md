
# Self-Driving Car Engineer Nanodegree

## Deep Learning

## Project: Behavioral Cloning

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/Image_00.png "NVIDIA Model"
[image2]: ./examples/stuck_track2.jpg "Stuck Image"
[image3]: ./examples/Image_02.png "Final Model"
[image4]: ./examples/track_01_center.jpg "track_01_center"
[image5]: ./examples/center_2016_12_01_13_43_52_882.jpg "center_2016_12_01_13_43_52_882.jpg"
[image6]: ./examples/center_2016_12_01_13_43_52_984.jpg "center_2016_12_01_13_43_52_984.jpg"
[image7]: ./examples/center_2016_12_01_13_43_53_084.jpg "center_2016_12_01_13_43_53_084.jpg"
[image8]: ./examples/center_2016_12_01_13_43_53_186.jpg "center_2016_12_01_13_43_53_186.jpg"
[image9]: ./examples/center_2016_12_01_13_43_53_287.jpg "center_2016_12_01_13_43_53_287.jpg"
[image10]: ./examples/center_2016_12_01_13_43_53_389.jpg "center_2016_12_01_13_43_53_389.jpg"

---
----
## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing a trained convolution neural network 
* `writeup_report.md` summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The `model.py` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.


---
### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

i used the same model, which was described by [NVIDIA](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) . 

The network consists of 9 layers:
- normalization layer, which uses a normalization formula: `(pixel/255) - 0.5` -  Refer to Keras lambda layer!
- 5 convolutional layers, which use strided convolutions in the first three convolutional layers with a 2×2 stride and a 5×5 kernel and a non-strided convolution with a 3×3 kernel size in the last two convolutional layers.
- 1 dropout layer to avoid data overfitting - with 50% Availability
- 1 flatten layer 
- 3 fully connected layers, which lead to an output control value which is the inverse turning radius.

![alt text][image1]

##### Model-Summary:
_________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    lambda_1 (Lambda)            (None, 66, 200, 3)        0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 31, 98, 24)        1824      
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 14, 47, 36)        21636     
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 5, 22, 48)         43248     
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 3, 20, 64)         27712     
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 1, 18, 64)         36928     
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 1, 18, 64)         0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 1152)              0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 100)               115300    
    _________________________________________________________________
    dense_2 (Dense)              (None, 50)                5050      
    _________________________________________________________________
    dense_3 (Dense)              (None, 10)                510       
    _________________________________________________________________
    dense_4 (Dense)              (None, 1)                 11        
    =================================================================
    Total params: 252,219
    Trainable params: 252,219
    Non-trainable params: 0
    _________________________________________________________________
    

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting 

The model was trained and validated on different data sets from Track-1/Track-2 to ensure that the model was not overfitting (Refer to data folder) - Images of the Track1/2 was not included in the Udacity-Workspace for space-issues. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 241).

    1. Number of epochs was set to 10, since the validation loss started to increase substantially, above that value. 
    2. Batch size was chosen to be 40, since it gave a smoother driving when compared to lower batch sizes of 16 or 32. 
    3. Number of augmented samples in each epoch were chosen to be 2x training data size - which is set to 20000 - this has already had made the trainning process took much-time.

    ------------------------------
    Parameters
    ------------------------------
    keep_prob            := 0.5
    nb_epoch             := 10
    samples_per_epoch    := 20000
    batch_size           := 40
    learning_rate        := 0.0001
    ------------------------------


#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. 
- I first used the sample data for track 1 (see the "Project Resources" lecture for the link), provided by Udacity.
- In order to provide more data - i implemented `data_generator` function (model.py line 122) - according to [REF](https://medium.com/@fromtheast/implement-fit-generator-in-keras-61aa2786ce98) that uses the following **Data-Augumentation Pipeline** :
    - Random Select one of the Images (Center,Left,Right) Images for certain Train-Point
    - Adjust Steering-Angle - based on the disccusion [Using Multiple Cameras](https://classroom.udacity.com/nanodegrees/nd013/parts/edf28735-efc1-4b99-8fbb-ba9c432239c8/modules/6b6c37bc-13a5-47c7-88ed-eb1fce9789a0/lessons/3fc8dd70-23b3-4f49-86eb-a8707f71f8dd/concepts/2cd424ad-a661-4754-8421-aec8cb018005) :
        - If Left: steering_angle += 0.2
        - If Right: steering_angle -= 0.2
        - If Center: Do Nothing
    - Randomly Flip Images and correcting the Steering-Angle -  based on the discussion [Using Data Augementation](https://classroom.udacity.com/nanodegrees/nd013/parts/edf28735-efc1-4b99-8fbb-ba9c432239c8/modules/6b6c37bc-13a5-47c7-88ed-eb1fce9789a0/lessons/3fc8dd70-23b3-4f49-86eb-a8707f71f8dd/concepts/580c6a1d-9d20-4d2e-a77d-755e0ca0d4cd)
    - Random Translate Images: adding artificial shifts and rotations helped to teach the network how to recover from a poor position or orientation.
    
- After that i added trainning data for track-2 in order to create more generalized model
- i noticed that the "loss" is decreasing very slowely - so i tuned `samples_per_epoch=20000` - so the model decrease the loss with keeping the learning-rate un-altered. 


---
### Architecture and Training Documentation

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was as decribed in the last section.

The final step was to run the simulator to see how well the car was driving around track one. After adding data from Track-2, i noticed that the vehicle was nearer to the left-lane rather than the center. 

At the end of the process, the vehicle is able to drive autonomously around the track-1 without leaving the road. (Refer to `run1.mp4`, saved in the workspace) 

At Track-2  (Refer to `run2.mp4`) the vehicle finished the 1st lap without any errors - and at the 2nd lap the vehicle curved with a small angle and ended up stuck in one of the road signs - as shown in the below figure.
![alt text][image2]

#### 2. Final Model Architecture

The model architecture was discussed in the previous section. Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)
![alt text][image3]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one while driving in the center. Here is an example image of center lane driving:
![alt text][image4]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to drive in center. These images show what a recovery train-images:

![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]

Then I repeated this process on track two in order to get more data points.

To augment the data set, I also flipped images and angles as mentioned within the Project-Discussion videos. The Augumentation was done on the fly - using `fit_generator()` as discussed earlier.

After the collection process, I had X number of data points. I then preprocessed this data (model.py line 51) by performing the following steps: 

- Crop-Images: i cropped 60 Pixels from Top (Sky) - 25 from Buttom (Car-Hood). I did not used Crop2D-Layer from Keras, since i encountered an Issue using the new Keras-API 2 - which i tried to avoid to meet the deadline.
- Resize-Image: I used OpenCV - `resize` function to reshape images into: (66,200,3)
- Convert the image from RGB to YUV - this was mentioned in [NVIDIA-Paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
- Shuffling the data was done by default by `fit_generator` from Keras.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. 
I used an adam optimizer so that manually training the learning rate wasn't necessary.

I used an iterative approach for the optimization of validation accuracy:
    1. Number of epochs was set to 10, since the validation loss started to increase substantially, above that value. 
    2. Batch size was chosen to be 40, since it gave a smoother driving when compared to lower batch sizes of 16 or 32. 
    3. Number of augmented samples in each epoch were chosen to be 2x training data size - which is set to 20000 - this has already had made the trainning process took much-time.


