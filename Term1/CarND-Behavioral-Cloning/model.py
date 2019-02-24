import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#Keras-Module
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Cropping2D
#visulations
from keras.utils import plot_model
import argparse
import cv2, os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# constants
IMG_HEIGHT = 66
IMG_WIDTH = 200
IMG_CHANNELS = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)


##########################################################################
### Image Pre-Processing Helper functions
##########################################################################
def load_image(data_dir, image_file):
    """
    Load RGB images from a file
    """
    return mpimg.imread(os.path.join(data_dir,image_file.strip()))

def crop(image):
    """
    Crop the image (removing the sky and the car hood)
    """
    return image[60:-25, :, :] # remove the sky and the car hood

def resize(image):
    """
    Resize the image to the input shape used by the network model
    """
    return cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT), cv2.INTER_AREA)

def rgb2yuv(image):
    """
    Convert the image from RGB to YUV, this is needed for NVIDIA model
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def preprocess_pipeline(image):
    """
    preprocess_pipeline
    """
    image = crop(image)
    image = resize(image)
    image = rgb2yuv(image)
    return image

##########################################################################
### Data Generation Helper-Functions
##########################################################################
def adjust_steering_angle(data_dir, center, left, right, steering_angle):
    """
    Randomly choose an image from the center, left or right, and adjust
    the steering angle.
    """
    choice = np.random.choice(3)
    if choice == 0:
        return load_image(data_dir, left), steering_angle + 0.2
    elif choice == 1:
        return load_image(data_dir, right), steering_angle - 0.2
    return load_image(data_dir, center), steering_angle


def random_flip(image, steering_angle):
    """
    Randomly flipt the image left <-> right, and adjust the steering angle.
    """
    #Avoid converting null - steering angle value
    if (np.random.rand() < 0.5) and (steering_angle != 0):
        image = np.fliplr(image)
        steering_angle = -steering_angle
    return image, steering_angle


def random_translate(image, steering_angle, range_x, range_y):
    """
    Randomly shift the image virtially and horizontally (translation).
    """
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle


def random_brightness(image):
    """
    Randomly adjust brightness of the image.
    """
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def data_augument(data_dir, center, left, right, steering_angle, range_x=100, range_y=10):
    """
    Generate an augumented image and adjust steering angle.
    (The steering angle is associated with the center image)
    """
    image, steering_angle = adjust_steering_angle(data_dir, center, left, right, steering_angle)
    image, steering_angle = random_flip(image, steering_angle)
    image, steering_angle = random_translate(image, steering_angle, range_x, range_y)
    image = random_brightness(image)
    return image, steering_angle
    

def data_generator(data_dir, image_paths, steering_angles, batch_size, is_training):
    """
    Generate training image give image paths and associated steering angles
    Ref: https://medium.com/@fromtheast/implement-fit-generator-in-keras-61aa2786ce98 -
         shows how to define generator function and use it in fit_generator()
    """
    #Create Empty Array to include Images/Output for Batches
    images = np.empty([batch_size, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])
    steers = np.empty(batch_size)
    # Loop forever so the generator never terminates
    while True:
        for i in range(batch_size):
            # choose random index in images
            index= np.random.choice(image_paths.shape[0])
            # extract center/left/right image-paths
            center, left, right = image_paths[index]
            # extract Steering Angle
            steering_angle = steering_angles[index]
            # argumentation
            if is_training and np.random.rand() < 0.6:
                image, steering_angle = data_augument(data_dir, center, left, right, steering_angle)
            else:
                image = load_image(data_dir, center) 
            # add the image and steering angle to the batch
            images[i] = preprocess_pipeline(image)
            steers[i] = steering_angle
        yield images, steers


def load_data(data_dir):
    """
    Load training data and split it into training and validation set
    """
    #reads CSV file into a single dataframe variable
    data_df = pd.read_csv(os.path.join(data_dir, 'driving_log.csv'), 
    names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])

    #yay dataframes, we can select rows and columns by their names
    #we'll store the camera images as our input data
    X = data_df[['center', 'left', 'right']].values[1:]
    #and our steering commands as our output data
    y = data_df['steering'].values[1:].astype(float)

    #now we can split the data into a training (80), testing(20), and validation set
    #thanks scikit learn
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, 
        test_size=0.2, random_state=0)

    return X_train, X_valid, y_train, y_valid


def build_model(keep_prob=0.5):
    """
    NVIDIA DAVE-2 - model used
The network consists of 10 layers:
- 1 normalization layer, which uses a normalization formula: `(pixel/255) - 0.5` -  Refer to Keras lambda layer!
- 5 convolutional layers, which use strided convolutions in the first three convolutional layers with a 2×2 stride and a 5×5 kernel and a non-strided convolution with a 3×3 kernel size in the last two convolutional layers.
- 1 dropout layer to avoid data overfitting - with 50% Availability
- 1 flatten layer 
- 3 fully connected layers, which lead to an output control value which is the inverse turning radius.
    """
    model = Sequential()
    #model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
    # normalization layer
    model.add(Lambda(lambda x: x/255-0.5, input_shape=INPUT_SHAPE))
    # Convolution-Layers
    model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    # Dropout-Layer
    model.add(Dropout(keep_prob))
    # Flatten-Layer
    model.add(Flatten())
    # Fully-Connected Layers
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    #print summary
    model.summary()
    # Save a Model-Plot
    plot_model(model, to_file=os.path.dirname(__file__) + '\\model.png')

    return model


def train_model(model, args, X_train, X_valid, y_train, y_valid):
    """
    Train the model
    """
    #Ref: https://keras.io/callbacks/
    # Saves the model after every epoch.
    # Arguments: 
    # - filepath: string, path to save the model file. 
    # - monitor: quantity to monitor. 
    # - verbose: verbosity mode, 0 or 1. 
    # - save_best_only: if save_best_only=True, the latest best model according to the quantity monitored 
    # - mode: one of {auto, min, max}. 
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='auto')

    #use Mean-Squared-Error for Loss/ Adam-Optimizer
    model.compile(loss='mean_squared_error', 
                  optimizer=Adam(lr=args.learning_rate))

    # compile and train the model using the generator function
    train_generator = data_generator(args.data_dir, X_train, y_train, args.batch_size, True)
    validation_generator = data_generator(args.data_dir, X_valid, y_valid, args.batch_size, False)

    #The generator is run in parallel to the model, for efficiency. 
    #For instance, this allows you to do real-time data augmentation on images on CPU in 
    #parallel to training your model on GPU.
    history_object = model.fit_generator(generator=train_generator,
                        samples_per_epoch= args.samples_per_epoch,
                        nb_epoch = args.nb_epoch,
                        max_q_size=1,
                        validation_data=validation_generator,
                        nb_val_samples=len(X_valid),
                        callbacks=[checkpoint],
                        verbose=1)
    
    #model.save('model.h5')
    return history_object


def main():
    """
    Load train/validation data set and train the model
    """
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', 
                        help='data directory',        
                        dest='data_dir',
                        type=str,   default='data')
    parser.add_argument('-k', 
                        help='drop out probability',  
                        dest='keep_prob',         
                        type=float, default=0.5)
    parser.add_argument('-n', 
                        help='number of epochs',      
                        dest='nb_epoch',          
                        type=int,   
                        default=10)
    parser.add_argument('-s', 
                        help='samples per epoch',     
                        dest='samples_per_epoch', 
                        type=int,   default=20000)
    parser.add_argument('-b', 
                        help='batch size',            
                        dest='batch_size',        
                        type=int,   default=40)
    parser.add_argument('-l', help='learning rate',         
                        dest='learning_rate',     
                        type=float, default=1.0e-4)
    args = parser.parse_args()

    #print parameters
    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    #load train-data
    args.data_dir = os.path.join(os.path.dirname(__file__), args.data_dir)
    # load-data
    data = load_data(args.data_dir)

    #build model
    model = build_model(args.keep_prob)
    #train model on data, it saves as model.h5 
    history = train_model(model, args, *data)

    # Save model weights
    model.save('model.h5')

    ### print the keys contained in the history object
    print(history.history.keys())

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('Loss - MSE')
    plt.xlabel('Epoch')
    plt.legend(['training set', 'validation set'], loc='upper left')
    plt.show()
    plt.savefig('training_validation.png')


if __name__ == '__main__':
    main()
