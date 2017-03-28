import pickle

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow.contrib.layers import flatten

def inception(x, drop):
    
    # Values for the random distribution of initial weights
    mu = 0
    sigma = 0.1

    # These parameters define the number of feature maps and the depth of the layers, and they are relatively 
    # small due to memory constraints on the training computer
    map1 = 6
    map2 = 12
    reduce1x1 = 3
    num_fc1 = 128
    num_fc2 = 43
    img_size = 32
    
    # This model follows loosely from the following article:
    # https://hacktilldawn.com/2016/09/25/inception-modules-explained-and-implemented/

    # Both inception modules have the following components:
    # Input -> 1x1 convolution -> concatenation
    # Input -> 1x1 convolution -> 3x3 convolution -> concatenation
    # Input -> 1x1 convolution -> 5x5 convolution -> concatenation
    # Input -> 3x3 max pooling -> 1x1 convolution -> concatenation
    
    # Inception module 1
    W_conv1_1x1_1 = tf.Variable(tf.truncated_normal(shape=(1,1,1,map1), mean = mu, stddev = sigma))
    b_conv1_1x1_1 = tf.Variable(tf.zeros(map1))
     
    W_conv1_1x1_2 = tf.Variable(tf.truncated_normal(shape=(1,1,1,reduce1x1), mean = mu, stddev = sigma))
    b_conv1_1x1_2 = tf.Variable(tf.zeros(reduce1x1))
     
    W_conv1_1x1_3 = tf.Variable(tf.truncated_normal(shape=(1,1,1,reduce1x1), mean = mu, stddev = sigma))
    b_conv1_1x1_3 = tf.Variable(tf.zeros(reduce1x1))
     
    W_conv1_3x3 = tf.Variable(tf.truncated_normal(shape=(3,3,reduce1x1,map1), mean = mu, stddev = sigma))
    b_conv1_3x3 = tf.Variable(tf.zeros(map1))
     
    W_conv1_5x5 = tf.Variable(tf.truncated_normal(shape=(5,5,reduce1x1,map1), mean = mu, stddev = sigma))
    b_conv1_5x5 = tf.Variable(tf.zeros(map1))
     
    W_conv1_1x1_4= tf.Variable(tf.truncated_normal(shape=(1,1,1,map1), mean = mu, stddev = sigma))
    b_conv1_1x1_4= tf.Variable(tf.zeros(map1))
    
    conv1_1x1_1 = tf.nn.conv2d(x, W_conv1_1x1_1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1_1x1_1
    conv1_1x1_2 = tf.nn.relu(tf.nn.conv2d(x, W_conv1_1x1_2, strides=[1, 1, 1, 1], padding='SAME') + b_conv1_1x1_2)
    conv1_1x1_3 = tf.nn.relu(tf.nn.conv2d(x, W_conv1_1x1_3, strides=[1, 1, 1, 1], padding='SAME') + b_conv1_1x1_3)
    conv1_3x3 = tf.nn.conv2d(conv1_1x1_2, W_conv1_3x3, strides=[1, 1, 1, 1], padding='SAME') + b_conv1_3x3
    conv1_5x5 = tf.nn.conv2d(conv1_1x1_3, W_conv1_5x5, strides=[1, 1, 1, 1], padding='SAME') + b_conv1_5x5
    maxpool1 = tf.nn.max_pool(x,ksize=[1,3,3,1],strides=[1,1,1,1],padding='SAME')
    conv1_1x1_4 = tf.nn.relu(tf.nn.conv2d(maxpool1, W_conv1_1x1_4, strides=[1, 1, 1, 1], padding='SAME') + b_conv1_1x1_4)
        
    # Concatenation of all feature maps, with ReLU
    inception1 = tf.nn.relu(tf.concat([conv1_1x1_1,conv1_3x3,conv1_5x5,conv1_1x1_4],3), name='inception1')

    #Inception Module2
    W_conv2_1x1_1 = tf.Variable(tf.truncated_normal(shape=(1,1,4*map1,map2), mean = mu, stddev = sigma))
    b_conv2_1x1_1 = tf.Variable(tf.zeros(map2))
     
    W_conv2_1x1_2 = tf.Variable(tf.truncated_normal(shape=(1,1,4*map1,reduce1x1), mean = mu, stddev = sigma))
    b_conv2_1x1_2 = tf.Variable(tf.zeros(reduce1x1))
     
    W_conv2_1x1_3 = tf.Variable(tf.truncated_normal(shape=(1,1,4*map1,reduce1x1), mean = mu, stddev = sigma))
    b_conv2_1x1_3 = tf.Variable(tf.zeros(reduce1x1))
     
    W_conv2_3x3 = tf.Variable(tf.truncated_normal(shape=(3,3,reduce1x1,map2), mean = mu, stddev = sigma))
    b_conv2_3x3 = tf.Variable(tf.zeros(map2))
     
    W_conv2_5x5 = tf.Variable(tf.truncated_normal(shape=(5,5,reduce1x1,map2), mean = mu, stddev = sigma))
    b_conv2_5x5 = tf.Variable(tf.zeros(map2))
     
    W_conv2_1x1_4= tf.Variable(tf.truncated_normal(shape=(1,1,4*map1,map2), mean = mu, stddev = sigma))
    b_conv2_1x1_4= tf.Variable(tf.zeros(map2))
        
    conv2_1x1_1 = tf.nn.conv2d(inception1, W_conv2_1x1_1, strides=[1, 1, 1, 1], padding='SAME') + b_conv2_1x1_1
    conv2_1x1_2 = tf.nn.relu(tf.nn.conv2d(inception1, W_conv2_1x1_2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2_1x1_2)
    conv2_1x1_3 = tf.nn.relu(tf.nn.conv2d(inception1, W_conv2_1x1_3, strides=[1, 1, 1, 1], padding='SAME') + b_conv2_1x1_3)
    conv2_3x3 = tf.nn.conv2d(conv2_1x1_2, W_conv2_3x3, strides=[1, 1, 1, 1], padding='SAME') + b_conv2_3x3
    conv2_5x5 = tf.nn.conv2d(conv2_1x1_3, W_conv2_5x5, strides=[1, 1, 1, 1], padding='SAME') + b_conv2_5x5
    maxpool2 = tf.nn.max_pool(inception1,ksize=[1,3,3,1],strides=[1,1,1,1],padding='SAME')
    conv2_1x1_4 = tf.nn.relu(tf.nn.conv2d(maxpool2, W_conv2_1x1_4, strides=[1, 1, 1, 1], padding='SAME') + b_conv2_1x1_4)
        
    # Concatenation of all feature maps, with ReLU
    inception2 = tf.nn.relu(tf.concat([conv2_1x1_1,conv2_3x3,conv2_5x5,conv2_1x1_4],3), name='inception2')

    # Flattening
    inception2_flat = tf.reshape(inception2,[-1,img_size*img_size*4*map2])
       
    # These are the two fully connected layers, the first of which has dropout when training
    W_fc1 = tf.Variable(tf.truncated_normal(shape=(img_size*img_size*(4*map2),num_fc1), mean = mu, stddev = sigma))
    b_fc1 = tf.Variable(tf.zeros(num_fc1))
        
    h_fc1 =tf.nn.dropout(tf.nn.relu(tf.matmul(inception2_flat,W_fc1)+b_fc1),drop)

    W_fc2 = tf.Variable(tf.truncated_normal(shape=(num_fc1,num_fc2), mean = mu, stddev = sigma))
    b_fc2 = tf.Variable(tf.zeros(num_fc2))
    
    logits = tf.matmul(h_fc1, W_fc2) + b_fc2
    
    return logits
    
def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, drop: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples
            
def normalize(x):
    return x/127.5 - 1.0
    
def grayscale(rgb):
    
    r, g, b = rgb[:,:,:,0], rgb[:,:,:,1], rgb[:,:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    
    return np.expand_dims(gray, axis=-1)
    
def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=3):
    # image_input: the test image being fed into the network to produce the feature maps
    # tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
    # activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
    # plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry
    
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input, drop: 1})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max)
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max)
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min)
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest")
    
## Importing data
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
    
## Visualizing a random image to ensure all is well
# Displaying a random sign from the training set, and printing its classification

# %matplotlib inline

[classes, idx, counts] = unique(y_train, return_index=True, return_counts=True)

plt.figure(num=1,figsize=(15,15))
plt.suptitle('Example of Each Classification from Training Set')
for i in range(1,44):
    image = X_train[idx[i-1]].squeeze()
    plt.subplot(5,9, i) # sets the number of feature maps to show on each row and column
    plt.title('Class. = ' + str(i-1)) # displays the feature map number
    plt.axis('off')
    plt.imshow(image)
    
plt.figure(num=2)
plt.bar(classes, counts)
plt.ylabel('Count')
plt.xlabel('Classification')
plt.title('Number of Training Examples for Each Classification')
plt.show()
    
## Preprocessing

# Shuffling the training data
X_train, y_train = shuffle(X_train, y_train)

# Normalizing data from (0,255) to (-1,1)
X_train = normalize(X_train)
X_valid = normalize(X_valid)
X_test = normalize(X_test)

# Converting to grayscale (Going from depth of 3 to 1)
X_train = grayscale(X_train)
X_valid = grayscale(X_valid)
X_test = grayscale(X_test)

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

## Setting up TensorFlow hyperparameters
EPOCHS = 15
BATCH_SIZE = 128

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
drop = tf.placeholder(tf.float32)
one_hot_y = tf.one_hot(y, 43)

rate = 0.001

logits = inception(x, drop)

testing_images = tf.nn.softmax(logits)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

## Training and saving the model
    
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("\nTraining...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, drop: 0.5})
            
        train_accuracy = evaluate(X_train, y_train)
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Train Accuracy = {:.3f}".format(train_accuracy))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './incept')
    print("Model saved")
    
## Testing with test data
with tf.Session() as sess:
    # saver.restore(sess, tf.train.latest_checkpoint('.'))
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, './incept')

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))

## Testing with web images    
strImageIn = ["outside_images/" + x for x in os.listdir("outside_images/")]
image_class = [28, 18, 27, 36, 33, 24, 14, 13]

plt.figure(num=3,figsize=(15,15))
plt.suptitle('Five Test Signs from the Web')
for i in range(1,9):
    X_web = mpimg.imread(strImageIn[i-1])
    image = X_web.squeeze()
    plt.subplot(2,4, i) # sets the number of feature maps to show on each row and column
    plt.title('Class. = ' + str(image_class[i-1])) # displays the feature map number
    plt.axis('off')
    plt.imshow(image)
    
with tf.Session() as sess:
    # saver.restore(sess, tf.train.latest_checkpoint('.'))
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, './incept')

    image_num = 0
    plt_num = 4
    successes = 0
    
    for image in strImageIn:
    
        X_test_map_img = mpimg.imread(image)
        X_test_map = np.expand_dims(X_test_map_img, axis=0)
        
        # Normalizing data from (0,255) to (-1,1)
        X_test_map = normalize(X_test_map)
        
        # Converting to grayscale (Going from depth of 3 to 1)
        X_test_map = grayscale(X_test_map)
        
        # Finding the softmax probabilities
        logits2 = sess.run(testing_images, feed_dict={x: X_test_map, drop: 1.0})
        
        # Getting the top five softmax results
        topKV = sess.run(tf.nn.top_k(logits2, k=5))
        
        # Reformatting the above topKV data to make plotting easier
        classes = topKV[1].squeeze();
        softs = topKV[0].squeeze();
        idx = np.argsort(classes)
        
        # Printing the above data as requested by Udacity
        print('\nTest image ' + str(image_num) + ' (true classification: ' + str(image_class[image_num]) + ')')
        print('Top five softmaxes: ' + str(softs))
        print('Corresponding classifications: ' + str(classes))
        
        Plotting bar graphs showing the top five softmax results for each of the 8 web images
        plt.figure(num=plt_num)
        plt.xticks(range(0,6), classes[idx])
        plt.bar(np.array(range(0,5)), softs[idx])
        plt.ylabel('Softmax')
        plt.xlabel('Neural Network Classification')
        plt.title('Top Five Softmax Probabilities for Test Image - True Classification: ' + str(image_class[image_num]))
        plt.show()
        
        # If we classified this web image correctly, we add it to a tally
        if classes[0] == image_class[image_num]:
            successes += 1
        
        image_num += 1
        plt_num += 1
        
    # Finding the accuracy over the web images
    web_image_accuracy = successes/8
    
    print('\nAccuracy over the images aquired on the web: ' + str(web_image_accuracy))

## Outputting example feature maps 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, './incept-complete/incept')

    # Outputting feature maps for both inception layers on a test image to visualize
    outputFeatureMap(X_test_map, sess.graph.get_tensor_by_name('inception1:0'), activation_min=-1, activation_max=-1 ,plt_num=plt_num)
    outputFeatureMap(X_test_map, sess.graph.get_tensor_by_name('inception2:0'), activation_min=-1, activation_max=-1 ,plt_num=plt_num+1)