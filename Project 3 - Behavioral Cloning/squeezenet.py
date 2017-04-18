from keras.applications.imagenet_utils import _obtain_input_shape
from keras import backend as K
from keras.layers import Input, Convolution2D, MaxPooling2D, Activation, concatenate, Dropout, GlobalAveragePooling2D, warnings, Dense, Flatten
from keras.models import Model, Sequential
from keras.layers import Cropping2D, Lambda, BatchNormalization
from keras.layers.advanced_activations import ELU
# from keras.engine.topology import get_source_inputs
# from keras.utils import get_file
# from keras.utils import layer_utils

sq1x1 = "squeeze1x1"
exp1x1 = "expand1x1"
exp3x3 = "expand3x3"
relu = "relu_"

WEIGHTS_PATH = "https://github.com/rcmalli/keras-squeezenet/releases/download/v1.0/squeezenet_weights_tf_dim_ordering_tf_kernels.h5"

# Modular function for Fire Node

def fire_module(x, fire_id, squeeze=16, expand=64):
    s_id = 'fire' + str(fire_id) + '/'

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3
    
    x = Convolution2D(squeeze, (1, 1), padding='valid', name=s_id + sq1x1)(x)
    x = Activation('relu', name=s_id + relu + sq1x1)(x)

    left = Convolution2D(expand, (1, 1), padding='valid', name=s_id + exp1x1)(x)
    left = Activation('relu', name=s_id + relu + exp1x1)(left)

    right = Convolution2D(expand, (3, 3), padding='same', name=s_id + exp3x3)(x)
    right = Activation('relu', name=s_id + relu + exp3x3)(right)

    x = concatenate([left, right], axis=channel_axis, name=s_id + 'concat')
    return x


# Original SqueezeNet from paper.

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
    
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
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
    
    
    # if weights not in {'imagenet', None}:
    #     raise ValueError('The `weights` argument should be either '
    #                      '`None` (random initialization) or `imagenet` '
    #                      '(pre-training on ImageNet).')

   ##   if weights == 'imagenet' and classes != 1000:
    #     raise ValueError('If using `weights` as imagenet with `include_top`'
    #                      ' as true, `classes` should be 1000')

   ##   img_input = Input(shape=(160,320,3))

   ##   ch, row, col = 3, 80, 320  # Trimmed image format
    # x = Cropping2D(cropping=((50,20), (0,0)))(img_input)
    # x = Lambda(lambda y: y/127.5 - 1., output_shape=(row, col, ch))(x)

   ##   x = Convolution2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv1')(img_input)
    # x = Activation('relu', name='relu_conv1')(x)
    # x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

   ##   x = fire_module(x, fire_id=2, squeeze=16, expand=64)
    # x = fire_module(x, fire_id=3, squeeze=16, expand=64)
    # x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

   ##   x = fire_module(x, fire_id=4, squeeze=32, expand=128)
    # x = fire_module(x, fire_id=5, squeeze=32, expand=128)
    # x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)

   ##   x = fire_module(x, fire_id=6, squeeze=48, expand=192)
    # x = fire_module(x, fire_id=7, squeeze=48, expand=192)
    # x = fire_module(x, fire_id=8, squeeze=64, expand=256)
    # x = fire_module(x, fire_id=9, squeeze=64, expand=256)
    # x = Dropout(0.5, name='drop9')(x)
    # 
    # x = Convolution2D(100, (1, 1), padding='valid', name='conv10')(x)
    # x = Activation('relu', name='relu_conv10')(x)
    # 
    # x = Flatten()(x)
    # x = Dense(128, name='dense_1')(x)
    # x = Activation('relu')(x)
    # 
    # x = Dense(64, name='dense_2')(x)
    # x = Activation('relu')(x)
    # 
    # x = Dense(16, name='dense_3')(x)
    # x = Activation('relu')(x)
    # 
    # out = Dense(1, name='dense_out')(x)
    # # x = GlobalAveragePooling2D()(x)
    # # out = Activation('softmax', name='loss')(x)

   ##   model = Model(img_input, out, name='squeezenet')

   ##   return model
