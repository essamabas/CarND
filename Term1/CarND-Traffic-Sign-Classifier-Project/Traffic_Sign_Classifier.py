
# Load pickled data
import pickle
import csv
import os


# TODO: Fill this in based on where you saved the training and testing data

training_file = "D:/carND/CarND-Traffic-Sign-Classifier-Project/data/train.p"
validation_file= "D:/carND/CarND-Traffic-Sign-Classifier-Project/data/valid.p"
testing_file = "D:/carND/CarND-Traffic-Sign-Classifier-Project/data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
x_train, y_train = train['features'], train['labels']
X_validation, Y_validation = valid['features'], valid['labels']
X_test, Y_test = test['features'], test['labels']



### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = len(x_train)

# TODO: Number of validation examples
n_validation = len(X_validation)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = x_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.

# Get all Labels
all_labels = []
#Select workbook
with open('D:/carND/CarND-Traffic-Sign-Classifier-Project/signnames.csv', 'r') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    # exclude 1st row 
    for row in readCSV:
        all_labels += [row[1]]
n_classes = len(all_labels)

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

import random
import numpy as np
import matplotlib.pyplot as plt


### Show images with it label.
num_of_samples=[]
plt.figure(figsize=(15, 30))
for i in range(0, n_classes):
    plt.subplot(9, 5, i+1)
    x_selected = x_train[y_train == i]
    plt.imshow(x_selected[0, :, :, :]) #draw the first image of each class
    plt.title(all_labels[i])
    plt.axis('off')
    num_of_samples.append(len(x_selected))
plt.show()


# histogram of label frequency
hist, bins = np.histogram(y_train, bins=n_classes)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.ylabel('Number of Images')
plt.xlabel('Classifier Number')
plt.show()

import cv2

# Generate Images for Learning
class ImageDataGenerator():
    
    # Init
    def __init__(self):
        pass
    
    @staticmethod
    def grayscale(images):
        image_list = []
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            image_list.append(gray)
        image_list = np.array(image_list)
        #image_list = np.reshape(image_list, (-1,32,32,1))
        return np.array(image_list)

    @staticmethod
    def normalize(images):
        output = images.astype(np.float32) * (1. / 255) - 0.5
        #output = (images - 128)/128
        return output
    
    # random translate
    @staticmethod
    def random_translate(images, labels, trans_range):
        # Check that Images/Labels have the same shape
        if(images.shape[0] != labels.shape[0]):
            raise Exception("Batch size Error.")
        
        # get width - height of 1st image
        img_width, img_height = images[0].shape
        
        # Convert degree to radian
        o_images = np.zeros_like(images)
        o_labels = labels
        
        # loop images
        for idx in range(images.shape[0]):
            # tranform          
            tr_x = trans_range * np.random.uniform() - trans_range / 2
            tr_y = trans_range * np.random.uniform() - trans_range / 2
            M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
            #o_images[idx] = np.expand_dims(cv2.warpAffine(images[idx],M,(img_width,img_height)), axis=2)
            o_images[idx] = cv2.warpAffine(images[idx],M,(img_width,img_height))
        
        return o_images, o_labels

    # random rotate images
    @staticmethod
    def random_rotate(images, labels, rot_range):
        
        # Check that Images/Labels have the same shape
        if(images.shape[0] != labels.shape[0]):
            raise Exception("Batch size Error.")
        
        # get width - height of 1st image
        img_width, img_height = images[0].shape
        
        # Convert degree to radian
        o_images = np.zeros_like(images)
        o_labels = labels
        
        # loop images
        for idx in range(images.shape[0]):
            # get random degree
            degree = np.random.uniform(-rot_range, rot_range)

            # tranform
            M = cv2.getRotationMatrix2D((img_width/2, img_height/2),degree,1)
            #o_images[idx] = np.expand_dims(cv2.warpAffine(images[idx],M,(img_width,img_height)), axis=2)
            o_images[idx] = cv2.warpAffine(images[idx],M,(img_width,img_height))

        return o_images, o_labels

from sklearn.utils import shuffle
#X_train, y_train = shuffle(X_train, y_train)

#from sklearn.model_selection import train_test_split
#X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.

My_ImageGenerator = ImageDataGenerator()
# normalize Images
#X_train = My_Batches.normalize(X_train)

x_train_gray = My_ImageGenerator.grayscale(x_train)
# Generate rotation Images
X_Rot, Y_Rot = My_ImageGenerator.random_rotate(x_train_gray, y_train, 30)
# Generate Translation Images
X_Tra, Y_Tra = My_ImageGenerator.random_translate(x_train_gray, y_train, 10)
# Append to Train-data Sets
X_train = np.append(x_train_gray, X_Rot, axis=0)
X_train = np.append(X_train, X_Tra, axis=0)
Y_train = np.append(y_train, [Y_Rot, Y_Tra])
# normalize Images
X_train = My_ImageGenerator.normalize(X_train)
X_train = np.reshape(X_train, (-1,32,32,1))



### Define your architecture here.
### Feel free to use as many code cells as needed.
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle
import os

class tf_model():
    # Init
    def __init__(self):
        self.model_path='D:/carND/CarND-Traffic-Sign-Classifier-Project/lenet.ckpt'

    # Add linear-Layer with relu-activation function
    @staticmethod
    def fc_layer(x, W, b, relu=True, name='fc'):
        with tf.name_scope(name):
            x = tf.matmul(x, W) + b
            tf.summary.histogram("weights", W)
            tf.summary.histogram("biases", b)
            if relu == True:
                act = tf.nn.relu(x)
                tf.summary.histogram("activations", act)
                return act
            else:
                return x

    # Add Convlution-2D with relu-activation function
    @staticmethod
    def conv2d(x, W, b, strides=1,padding='VALID',name='conv'):
        with tf.name_scope(name):
            conv = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
            conv = tf.nn.bias_add(conv, b)
            conv  = tf.nn.relu(conv)
            tf.summary.histogram("weights",W)
            tf.summary.histogram("bias",b)
            tf.summary.histogram("activations",conv)
            return conv

    @staticmethod
    def maxpool2d(x, k=2,padding='VALID'):
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding=padding)

    @staticmethod
    def LeNet(x, keep_prob, mu = 0, sigma = 0.1):    

        # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
        # mu: mean
        # stddev: standard-deviation

        # Store layers weight & bias
        weights = {
            'wc1': tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma), name="W"),
            'wc2': tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma), name="W"),
            'fc1': tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma),name="W"),
            'fc2': tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma),name="W"),
            'out': tf.Variable(tf.truncated_normal(shape=(84, n_classes), mean = mu, stddev = sigma),name="W")}

        biases = {
            'bc1': tf.Variable(tf.random_normal([6]), name="B"),
            'bc2': tf.Variable(tf.random_normal([16]),name="B"),
            'fc1': tf.Variable(tf.random_normal([120]),name="B"),
            'fc2': tf.Variable(tf.random_normal([84]),name="B"),
            'out': tf.Variable(tf.random_normal([n_classes]),name="B")}

        # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
        # Activation.
        conv1 = tf_model.conv2d(x, weights['wc1'], biases['bc1'],strides=1,padding='VALID',name='conv1')
        # Pooling. Input = 28x28x6. Output = 14x14x6.
        conv1 = tf_model.maxpool2d(conv1, k=2)

        # Layer 2: Convolutional. Output = 10x10x16.
        # Activation.
        conv2 = tf_model.conv2d(conv1, weights['wc2'], biases['bc2'],strides=1,padding='VALID',name='conv2')
        # Pooling. Input = 10x10x16. Output = 5x5x16.
        conv2 = tf_model.maxpool2d(conv2, k=2)

        # Flatten. Input = 5x5x16. Output = 400.
        fc0   = flatten(conv2)

        # Layer 3: Fully Connected. Input = 400. Output = 120.
        # Activation.
        fc1 = tf_model.fc_layer(fc0, weights['fc1'], biases['fc1'],name='fc1')
        
        # drop out to prevent overfitting
        fc1 = tf.nn.dropout(fc1, keep_prob)

        # Layer 4: Fully Connected. Input = 120. Output = 84.
        # Activation.
        fc2 = tf_model.fc_layer(fc1, weights['fc2'], biases['fc2'], name='fc2')
        
        # drop out to prevent overfitting
        fc2 = tf.nn.dropout(fc2, keep_prob)

        # Layer 5: Fully Connected. Input = 84. Output = 10.
        logits = tf_model.fc_layer(fc2, weights['out'], biases['out'], relu=False,  name='out')
        
        return logits

    def loss_optimizer(self, logits, one_hot_y, learning_rate=0.001):
        
        # Loss & Optimization
        with tf.name_scope("loss_operation"):
            self.loss_operation = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=logits, labels=one_hot_y), name="loss_operation")
            tf.summary.scalar("loss_operation", self.loss_operation)

        with tf.name_scope("training_operation"):
            self.training_operation = tf.train.AdamOptimizer(learning_rate).minimize(self.loss_operation)

        with tf.name_scope("accuracy_operation"):
            self.correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
            self.accuracy_operation = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            tf.summary.scalar("accuracy_operation", self.accuracy_operation)

        self.summ = tf.summary.merge_all()

        # Save Model
        self.saver = tf.train.Saver()

        return self.loss_operation, self.training_operation, self.accuracy_operation
    
    # Evaluate Model
    def evaluate(self, X_data, y_data, BATCH_SIZE = 128):
        num_examples = len(X_data)
        total_accuracy = 0
        # get default session
        sess = tf.get_default_session()
        # 
        for offset in range(0, num_examples, BATCH_SIZE):
            batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
            # use accuracy
            accuracy = sess.run(self.accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples
    
    '''Predict Classifications
       parameters: 
       - images: array of images to be predicted
       - logits: extracted from Model
    '''
    def predict(self, images, logits, y_data=[], BATCH_SIZE = 128):
        # initial accuracy
        test_accuracy = 0.0
        predictions = []
        predictionIndicies = []

        #start session
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # Restore variables from disk.
            self.saver.restore(sess, self.model_path)
            print("Model restored.")
            softmax = tf.nn.softmax(logits)
            pred = tf.nn.top_k(softmax, 5)
            # if y_data are passed - then check accuracy
            if len(y_data)>0:
                test_accuracy = self.evaluate(images, y_data, BATCH_SIZE)
            result = sess.run(pred, feed_dict={x: images, keep_prob: 1.0})
            predictions  = result.values
            predictionIndicies  = result.indices
            
        return predictions, predictionIndicies, test_accuracy
    

    def train(self, X_train, Y_train, EPOCHS = 10, BATCH_SIZE = 128, name=""):

        # Create Session
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            num_examples = len(X_train)

            print("Training...")
            print()

            # Write Summary for Tensorboard
            LOGDIR = os.getcwd() + "/log/"
            writer = tf.summary.FileWriter(LOGDIR + name)
            merged_summary = tf.summary.merge_all()
            writer.add_graph(sess.graph)

            X_train, Y_train = shuffle(X_train, Y_train)

            # loop in Epochs
            for i in range(EPOCHS):
                for offset in range(0, num_examples, BATCH_SIZE):
                    end = offset + BATCH_SIZE
                    batch_x, batch_y = X_train[offset:end], Y_train[offset:end]
                    sess.run(self.training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.7})
                    # measuare accuarcy for every 5-Epochs
                    if i % 2 == 0:
                        [train_accuracy, s] = sess.run([self.accuracy_operation, merged_summary], 
                                                       feed_dict={x: batch_x, y: batch_y})
                        writer.add_summary(s, i)

                validation_accuracy = self.evaluate(X_validation, Y_validation, BATCH_SIZE)
                print("EPOCH {} ...".format(i+1))
                print("Validation Accuracy = {:.3f}".format(validation_accuracy))
                print()


            # Save Model
            save_path = self.saver.save(sess, self.model_path)
            print('Run `tensorboard --logdir=%s` to see the results.' % LOGDIR)
            print("Model saved")
            
            return train_accuracy, validation_accuracy

x = tf.placeholder(tf.float32, (None, 32, 32, 1), name="x")
y = tf.placeholder(tf.int32, (None), name="labels")
keep_prob = tf.placeholder(tf.float32) # probability to keep units
one_hot_y = tf.one_hot(y, n_classes)

### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.

def make_hparam_string(learning_rate,epoch,batch_size):
    tmp_str = "lr_%.0E_epochs_%d_batchsize_%d" % (learning_rate, epoch, batch_size)
    print(tmp_str)
    return tmp_str

# You can try adding some more learning rates
for learning_rate in [1E-3]:
    for epoch in [10]:
        for batch_size in [128]:
            # Make instance of TF-Model Class
            My_Model = tf_model()
            # Get LeNet-Model
            logits = My_Model.LeNet(x, keep_prob)
            # Define Loss-Optimizer
            My_Model.loss_optimizer(logits, one_hot_y, learning_rate)
            # Train the Model
            train_accuracy, validation_accuracy = My_Model.train(X_train, Y_train, 
                                                                 EPOCHS = epoch, BATCH_SIZE = batch_size, 
                                                                 name=make_hparam_string(learning_rate,epoch,batch_size))




### Load the images and plot them here.
### Feel free to use as many code cells as needed.
### Load the images and plot them here.
### Feel free to use as many code cells as needed.
import os
import cv2
import matplotlib.image as mpimg
web_images = os.listdir("web_images/")
test_labels = np.array([33,5,0,14,40])

# Show the images, add to a list to process for classifying
test_images = []
for i in web_images:
    i = 'web_images/' + i
    print(i)
    image = cv2.imread(i)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (32,32))
    test_images.append(image)
    plt.figure(figsize=(1,5))
    plt.imshow(image)
    plt.show()

### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.
predictions, predictionIndicies, test_accuracy = My_Model.predict(test_images,logits,test_labels)

### Calculate the accuracy for these 5 new images. 
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.
print("Test Accuracy = {:.3f}".format(test_accuracy))

### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 
### Feel free to use as many code cells as needed.

### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")
