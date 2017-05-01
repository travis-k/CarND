from keras.models import Sequential, Model
from keras.layers import Cropping2D, Lambda, Flatten, Dense, Activation
import matplotlib.pyplot as plt
import cv2
import os
import csv
from nn_arch import nn_arch
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import keras
print(keras.__version__)

def generator(samples, bias = 0.7, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:

                # Chosing randomly whether center, left or right
                img_choice = np.random.randint(3) 
                name = 'IMG_mine\\' + batch_sample[img_choice].split('\\')[-1]             
                img = cv2.imread(name)
                angle = float(batch_sample[3])
                
                # Correction for if we choose one of the side cameras
                correction = 0.25
                if img_choice == 1: # Left camera
                    angle += correction
                elif img_choice == 2: # Right camera
                    angle -= correction
                    
                # Randomly flipping sometimes
                if np.random.randint(2) == 0:
                    img = np.fliplr(img)
                    angle = -angle
                
                # Crude measure to reduce zero angle bias
                threshold = np.random.uniform()
                if (abs(angle) + bias) >= threshold or angles == []:
                    images.append(img)
                    angles.append(angle)
                
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
            
## Creating training/validation sets

samples = []
# with open('./driving_log.csv') as csvfile:
with open('./driving_log_mine.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=8)
validation_generator = generator(validation_samples, batch_size=8)

## Compiling the model 

model = nn_arch()
model.compile(loss='mse', optimizer='adam')
# model.load_weights('model_weights.h5')

## Train the model
history_object = model.fit_generator(train_generator, steps_per_epoch=200, validation_data=validation_generator, validation_steps=200, epochs=6, verbose=1)

model.save_weights('model_weights.h5')
model.save('model.h5')

## print the keys contained in the history object
print(history_object.history.keys())
            
## plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()