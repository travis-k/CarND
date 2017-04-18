#**Behavioral Cloning** 

##Writeup
---

**Behavioral Cloning Project**

NOTE: This project used 23,938 training images in a local folder "IMG". These images are not included on the project's Git, because of the volume. Please contact me at tkrebs@ryerson.ca for these images.

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./images/figure_1.png "Example training data and steering angles."
[image2]: ./images/figure_2.png "Histogram of steering angles with and without augmentation and modification."
[image3]: ./images/cnn-architecture-624x890.png "NVIDIA CNN architecture."
[image4]: ./images/model.png "My model architecture"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

As noted above, the folder IMG was not included on the Git, as it is too large. Please contact me if you need these files (tkrebs@ryerson.ca). 

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model is based on the NVIDIA architecture, but some of the layers are reduced in size due to memory constraints. This architecture is, I believe, well suited for this application.

By altering the sizes of the layers, I was able to run this network on my local graphics card. Each epoch would take approximately a minute. This network was fast and allowed for low losses for this application.

####2. Attempts to reduce overfitting in the model

I used dropout layers 2 times throughout the architecture, to try to avoid overfitting the data. I also randomly flipped the images, as described in Section 3.

The training procedure used a 80/20 train/validation split to ensure overfitting wasn't occuring. The validation loss was typically equal to or below the training loss, which makes sense to me given the dropout.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. The correction for the left/right camera steering angle was experimentally found to work at 0.25, so this was the value that was kept. 

The batch size from the generators was kept low, at 8, due to the size of the network and the fact that I was running it on my own GPU, not an AWS instance. 

####4. Appropriate training data

Most of my training data was with the car centered in the lane, but the data also included plenty of recovery maneuvers from the left and ride sides of the road to teach the car to recenter itself.

###Model Architecture and Training Strategy

####1. Solution Design Approach

I began with the NVIDIA CNN architecture, and data which I created myself. My losses in the beginning were high, and by introducing cropping and batch normalization, I helped to improve the model greatly.

The cropping really helped to improve not only the speed of the neural network training, but also the autonomous driving.

The layer sizes for my NN are smaller than NVIDIA, as I have memory constraints to worry about (on my laptop). 

Everything is dealt with in the model, so a raw RGB photo is the model input. I do not implement grayscale, though I should consider adding it in the future. 

####2. Final Model Architecture

The final model architecture follows the NVIDIA CNN architecture, with 4 convolution layers, a max pooling layer, and 4 fully connected layers. I believe this is a standard architecture for this application.

The standard NVIDIA CNN architecture is shown here.
![alt text][image3]

My modified architecture, to reduce memory requirements, is shown below.
* Input
* Cropping
* Normalization 
* 3x3 Convolution (32 maps)
* 3x3 Convolution (16 maps)
* 3x3 Convolution (8 maps)
* 3x3 Convolution (4 maps)
* 2x2 Max pooling with 25% dropout
* Fully connected layer (64 outputs) with 20% dropout
* Fully connected layer (32 outputs)
* Fully connected layer (16 outputs)
* Fully connected layer (1 output)

![alt text][image4]

The summary from Keras states:

*Total params: 1,447,677
*Trainable params: 1,447,517
*Non-trainable params: 160

####3. Creation of the Training Set & Training Process

To create training data, I performed three laps of Track 1 while staying in the center of the road, so the model could learn good behaviour. I then performed roughly 40 recovery maneouvers from the sides of the road, to teach the car to recover to the center if it happens to go off course. I made sure to only start recording when the car was off to the center, so I did not teach it to go off center.

![alt text][image1]

I randomly flipped both the images and steering angles, to eliminate the left-turn bias gained from Track 1. 

I also reduced the zero angle bias by crudly reducing the amount of zero angle examples passed into the training set, using a bias to pass over some of these examples. This can be seen in the generator function in model.py.

Also, to aid in recovery, I randomly select center, left or right camera for each snapshot. I apply a correction to the angle if it choses a side camera. This can also be seen in the generator in model.py.

![alt text][image2]

###Future Considerations

The trained model, as is, will go around Track 1 but barely. I believe the fragile nature of the model is due to the training data, which is some 23,000+ images not including data augmentation. I believe cleaner, higher quality training data would improve the model, but it is difficult to achieve with a mouse and keyboard on the Udacity simulator. 

The model could also be improved by augmenting the data with translations and alterations to the brightness. 

The model will likely not work (at all) on Track 2. Training with the second track may help generalize the model. I also believe the hills and light/shade patches of Track 2 would require a much more diverse training set. 
