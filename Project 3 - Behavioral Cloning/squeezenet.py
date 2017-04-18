from keras.applications.imagenet_utils import _obtain_input_shape
from keras import backend as K
from keras.layers import Input, Convolution2D, MaxPooling2D, Activation, concatenate, Dropout, GlobalAveragePooling2D, warnings, Dense, Flatten
from keras.models import Model, Sequential
from keras.layers import Cropping2D, Lambda, BatchNormalization
from keras.layers.advanced_activations import ELU

def squeezenet():
    
    input_shape=(160,320,3)
    nb_classes = 1
    
    # If model and weights do not exist in the local folder,
    # initiate a model
    
    # number of convolutional filters to use
    nb_filters1 = 32
    nb_filters2 = 16
    nb_filters3 = 8
    nb_filters4 = 4
    
    # size of pooling area for max pooling
    pool_size = (2, 2)
    
    # convolution kernel size
    kernel_size = (3, 3)
    
    # Initiating the model
    model = Sequential()
    
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=input_shape))
    # model.add(Lambda(lambda x: (x / 255.0) - 0.5))  
    model.add(BatchNormalization(axis=1)) 
    
    # Starting with the convolutional layer
    # The first layer will turn 1 channel into 16 channels
    model.add(Convolution2D(nb_filters1, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=input_shape))
    # Applying ReLU
    model.add(Activation('relu'))
    # Apply dropout of 50%
    model.add(Dropout(0.5))
    # The second conv layer will convert 16 channels into 8 channels
    model.add(Convolution2D(nb_filters2, kernel_size[0], kernel_size[1]))
    # Applying ReLU
    model.add(Activation('relu'))
    # The second conv layer will convert 8 channels into 4 channels
    model.add(Convolution2D(nb_filters3, kernel_size[0], kernel_size[1]))
    # Applying ReLU
    model.add(Activation('relu'))
    # The second conv layer will convert 4 channels into 2 channels
    model.add(Convolution2D(nb_filters4, kernel_size[0], kernel_size[1]))
    # Applying ReLU
    model.add(Activation('relu'))
    # Apply Max Pooling for each 2 x 2 pixels
    model.add(MaxPooling2D(pool_size=pool_size))
    # Apply dropout of 25%
    model.add(Dropout(0.25))
    
    # Flatten the matrix. The input has size of 360
    model.add(Flatten())
    # Input 360 Output 16
    model.add(Dense(64))
    # Applying ReLU
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    # Input 16 Output 16
    model.add(Dense(32))
    # Applying ReLU
    model.add(Activation('relu'))
    # Input 16 Output 16
    model.add(Dense(16))
    # Applying ReLU
    model.add(Activation('relu'))
    # Input 16 Output 1
    model.add(Dense(nb_classes))
    
    return(model)