from keras.applications.imagenet_utils import _obtain_input_shape
from keras import backend as K
from keras.layers import Input, Convolution2D, MaxPooling2D, Activation, concatenate, Dropout, GlobalAveragePooling2D, warnings, Dense, Flatten
from keras.models import Model, Sequential
from keras.layers import Cropping2D, Lambda, BatchNormalization
from keras.layers.advanced_activations import ELU

def nn_arch():
    
    input_shape=(160,320,3)

    model = Sequential()
    
    # Cropping the image to get rid of everything above the horizon, and the hood of the car
    model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=input_shape))

    # Using the lambda function was giving me write errors later on, so I use
    # BatchNormalization to do the same thing (normalize RGB)
    model.add(BatchNormalization(axis=1)) 
    
    # These are the four 3x3 convolutions, with decreasing feature maps
    model.add(Convolution2D(32, (3,3), border_mode='valid',input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(16, (3,3)))
    model.add(Activation('relu'))
    model.add(Convolution2D(8, (3,3)))
    model.add(Activation('relu'))
    model.add(Convolution2D(4, (3,3)))
    model.add(Activation('relu'))
    
    # 2x2 max pooling
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    
    # The three fully connected end layers, of decreasing size
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))

    # The output
    model.add(Dense(1))
    
    return(model)