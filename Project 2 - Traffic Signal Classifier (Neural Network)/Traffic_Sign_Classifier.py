import pickle

import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow.contrib.layers import flatten

def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # SOLUTION: Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)

    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)
    
    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # SOLUTION: Activation.
    fc2    = tf.nn.relu(fc2)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 43.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits

    ## Inception
def inception(x, train):
    
    mu = 0
    sigma = 0.1

    map1 = 6
    map2 = 12
    reduce1x1 = 3
    num_fc1 = 350
    num_fc2 = 43
    dropout=0.7
    img_size = 32
    
    #https://hacktilldawn.com/2016/09/25/inception-modules-explained-and-implemented/

    # Input -> 1x1 convolution -> concatenation
    # Input -> 1x1 convolution -> 3x3 convolution -> concatenation
    # Input -> 1x1 convolution -> 5x5 convolution -> concatenation
    # Input -> 3x3 average pooling -> 1x1 convolution -> concatenation

    # tf.Variable(tf.truncated_normal(shape=(1, 1, 1, 6), mean = mu, stddev = sigma))
    
    # Inception module 1

    W_conv1_1x1_1 = tf.Variable(tf.truncated_normal(shape=(1,1,3,map1), mean = mu, stddev = sigma))
    b_conv1_1x1_1 = tf.Variable(tf.zeros(map1))
     
    #follows input
    W_conv1_1x1_2 = tf.Variable(tf.truncated_normal(shape=(1,1,3,reduce1x1), mean = mu, stddev = sigma))
    b_conv1_1x1_2 = tf.Variable(tf.zeros(reduce1x1))
     
    #follows input
    W_conv1_1x1_3 = tf.Variable(tf.truncated_normal(shape=(1,1,3,reduce1x1), mean = mu, stddev = sigma))
    b_conv1_1x1_3 = tf.Variable(tf.zeros(reduce1x1))
     
    #follows 1x1_2
    W_conv1_3x3 = tf.Variable(tf.truncated_normal(shape=(3,3,reduce1x1,map1), mean = mu, stddev = sigma))
    b_conv1_3x3 = tf.Variable(tf.zeros(map1))
     
    #follows 1x1_3
    W_conv1_5x5 = tf.Variable(tf.truncated_normal(shape=(5,5,reduce1x1,map1), mean = mu, stddev = sigma))
    b_conv1_5x5 = tf.Variable(tf.zeros(map1))
     
    #follows max pooling
    W_conv1_1x1_4= tf.Variable(tf.truncated_normal(shape=(1,1,3,map1), mean = mu, stddev = sigma))
    b_conv1_1x1_4= tf.Variable(tf.zeros(map1))
    

    conv1_1x1_1 = tf.nn.conv2d(x, W_conv1_1x1_1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1_1x1_1
    conv1_1x1_2 = tf.nn.relu(tf.nn.conv2d(x, W_conv1_1x1_2, strides=[1, 1, 1, 1], padding='SAME') + b_conv1_1x1_2)
    conv1_1x1_3 = tf.nn.relu(tf.nn.conv2d(x, W_conv1_1x1_3, strides=[1, 1, 1, 1], padding='SAME') + b_conv1_1x1_3)
    conv1_3x3 = tf.nn.conv2d(conv1_1x1_2, W_conv1_3x3, strides=[1, 1, 1, 1], padding='SAME') + b_conv1_3x3
    conv1_5x5 = tf.nn.conv2d(conv1_1x1_3, W_conv1_5x5, strides=[1, 1, 1, 1], padding='SAME') + b_conv1_5x5
    maxpool1 = tf.nn.avg_pool(x,ksize=[1,3,3,1],strides=[1,1,1,1],padding='SAME')
    conv1_1x1_4 = tf.nn.relu(tf.nn.conv2d(maxpool1, W_conv1_1x1_4, strides=[1, 1, 1, 1], padding='SAME') + b_conv1_1x1_4)
    
        
    #concatenate all the feature maps and hit them with a relu
    inception1 = tf.nn.relu(tf.concat([conv1_1x1_1,conv1_3x3,conv1_5x5,conv1_1x1_4],3))

    #Inception Module2

    #follows inception1
    W_conv2_1x1_1 = tf.Variable(tf.truncated_normal(shape=(1,1,4*map1,map2), mean = mu, stddev = sigma))
    b_conv2_1x1_1 = tf.Variable(tf.zeros(map2))
     
    #follows inception1
    W_conv2_1x1_2 = tf.Variable(tf.truncated_normal(shape=(1,1,4*map1,reduce1x1), mean = mu, stddev = sigma))
    b_conv2_1x1_2 = tf.Variable(tf.zeros(reduce1x1))
     
    #follows inception1
    W_conv2_1x1_3 = tf.Variable(tf.truncated_normal(shape=(1,1,4*map1,reduce1x1), mean = mu, stddev = sigma))
    b_conv2_1x1_3 = tf.Variable(tf.zeros(reduce1x1))
     
    #follows 1x1_2
    W_conv2_3x3 = tf.Variable(tf.truncated_normal(shape=(3,3,reduce1x1,map2), mean = mu, stddev = sigma))
    b_conv2_3x3 = tf.Variable(tf.zeros(map2))
     
    #follows 1x1_3
    W_conv2_5x5 = tf.Variable(tf.truncated_normal(shape=(5,5,reduce1x1,map2), mean = mu, stddev = sigma))
    b_conv2_5x5 = tf.Variable(tf.zeros(map2))
     
    #follows max pooling
    W_conv2_1x1_4= tf.Variable(tf.truncated_normal(shape=(1,1,4*map1,map2), mean = mu, stddev = sigma))
    b_conv2_1x1_4= tf.Variable(tf.zeros(map2))
        
    #Inception Module 2
    conv2_1x1_1 = tf.nn.conv2d(inception1, W_conv2_1x1_1, strides=[1, 1, 1, 1], padding='SAME') + b_conv2_1x1_1
    conv2_1x1_2 = tf.nn.relu(tf.nn.conv2d(inception1, W_conv2_1x1_2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2_1x1_2)
    conv2_1x1_3 = tf.nn.relu(tf.nn.conv2d(inception1, W_conv2_1x1_3, strides=[1, 1, 1, 1], padding='SAME') + b_conv2_1x1_3)
    conv2_3x3 = tf.nn.conv2d(conv2_1x1_2, W_conv2_3x3, strides=[1, 1, 1, 1], padding='SAME') + b_conv2_3x3
    conv2_5x5 = tf.nn.conv2d(conv2_1x1_3, W_conv2_5x5, strides=[1, 1, 1, 1], padding='SAME') + b_conv2_5x5
    maxpool2 = tf.nn.avg_pool(inception1,ksize=[1,3,3,1],strides=[1,1,1,1],padding='SAME')
    conv2_1x1_4 = tf.nn.relu(tf.nn.conv2d(maxpool2, W_conv2_1x1_4, strides=[1, 1, 1, 1], padding='SAME') + b_conv2_1x1_4)
        
    #concatenate all the feature maps and hit them with a relu
    inception2 = tf.nn.relu(tf.concat([conv2_1x1_1,conv2_3x3,conv2_5x5,conv2_1x1_4],3))

    #flatten features for fully connected layer
    inception2_flat = tf.reshape(inception2,[-1,img_size*img_size*4*map2])
       
    #Fully connected layers
    #since padding is same, the feature map with there will be 4 28*28*map2
    W_fc1 = tf.Variable(tf.truncated_normal(shape=(img_size*img_size*(4*map2),num_fc1), mean = mu, stddev = sigma))
    b_fc1 = tf.Variable(tf.zeros(num_fc1))
    
    W_fc2 = tf.Variable(tf.truncated_normal(shape=(num_fc1,num_fc2), mean = mu, stddev = sigma))
    b_fc2 = tf.Variable(tf.zeros(num_fc2))
        
    #Fully connected layers
    if train:
        h_fc1 =tf.nn.dropout(tf.nn.relu(tf.matmul(inception2_flat,W_fc1)+b_fc1),dropout)
    else:
        h_fc1 = tf.nn.relu(tf.matmul(inception2_flat,W_fc1)+b_fc1)

    logits = tf.matmul(h_fc1, W_fc2) + b_fc2
    
    return logits
    
def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples
    
def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it maybe having trouble accessing the variable from inside a function
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

## Importing data/preprocessing
# All images are 32x32, so no resizing is necessary

# Loading data provided by Udacity
training_file = 'sign_data/train.p'
validation_file='sign_data/valid.p'
testing_file = 'sign_data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

# Shuffling the training data
X_train, y_train = shuffle(X_train, y_train)

# Ensuring there are no issues with data sizes
assert(len(X_train) == len(y_train))
assert(len(X_valid) == len(y_valid))
assert(len(X_test) == len(y_test))

## Dataset summary
# Information about the data set
n_train = len(X_train)
n_valid = len(X_valid)
n_test = len(X_test)
image_shape = X_train[0].shape
n_classes = max(y_train)

print("\nPreprocessing:")
print("Number of training examples =", n_train)
print("Number of validation examples =", n_valid)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

## Visualizing a random image to ensure all is well
# Displaying a random sign from the training set, and printing its classification

# %matplotlib inline
# 
# index = random.randint(0, len(X_train))
# image = X_train[index].squeeze()
# 
# plt.figure(figsize=(1,1))
# plt.imshow(image)
# print("\nRandom image visualization:")
# print("Classification = ", y_train[index])

## Setting up TensorFlow hyperparameters
EPOCHS = 100
BATCH_SIZE = 128

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)

rate = 0.0001

# logits = LeNet(x)
logits = inception(x, train=False)
# logits_notrain = inception(x, train=False)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
# optimizer = tf.train.AdamOptimizer(learning_rate = rate)
optimizer = tf.train.AdagradOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

# correct_prediction = tf.equal(tf.argmax(logits_notrain, 1), tf.argmax(one_hot_y, 1))
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

## Training and saving the model
    
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")
    
# with tf.Session() as sess:
#     saver.restore(sess, tf.train.latest_checkpoint('.'))
# 
#     test_accuracy = evaluate(X_test, y_test)
#     print("Test Accuracy = {:.3f}".format(test_accuracy))

### Preprocess the data here. Preprocessing steps could include normalization, converting to grayscale, etc.
### Feel free to use as many code cells as needed.

### Define your architecture here.
### Feel free to use as many code cells as needed.

### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.

### Load the images and plot them here.
### Feel free to use as many code cells as needed.

### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.

### Calculate the accuracy for these 5 new images. 
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.

### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 
### Feel free to use as many code cells as needed.

### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

