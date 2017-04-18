#**Behavioral Cloning** 

##Writeup
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

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

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model is based on the NVIDIA architecture,

* Input
* Cropping
* Normalization 
* 3x3 Convolution (32 maps) with 50% dropout
* 3x3 Convolution (16 maps)
* 3x3 Convolution (8 maps)
* 3x3 Convolution (4 maps)
* 2x2 Max pooling with 25% dropout
* Fully connected layer (64 outputs) with 20% dropout
* Fully connected layer (32 outputs)
* Fully connected layer (16 outputs)
* Fully connected layer (1 output)

####2. Attempts to reduce overfitting in the model

I used dropout layers 3 times throughout the architecture, to try to avoid overfitting the data. I also randomly flipped the images, as described in Section 3. 

The training procedure used a 80/20 train/validation split to ensure overfitting wasn't occuring. 

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. The correction for the left/right camera steering angle was experimentally found to work at 0.25, so this was the value that was kept. 

The batch size from the generators was kept low, at 8, due to the size of the network and the fact that I was running it on my own GPU, not an AWS instance. 

####4. Appropriate training data

Most of my training data was with the car centered in the lane, but the data also included plenty of recovery maneuvers from the left and ride sides of the road to teach the car to recenter itself.


###Model Architecture and Training Strategy

####1. Solution Design Approach

I began with the NVIDIA CNN architecture, and data which I created myself. My losses in the beginning were high, and by introducing cropping and batch normalization, I helped to improve the model greatly.

The layer sizes for my NN are smaller than NVIDIA, as I have memory constraints to worry about (on my laptop). 

Everything is dealt with in the model, so a raw RGB photo is the model input. I do not implement grayscale, though I should consider adding it in the future. 


####2. Final Model Architecture

The final model architecture follows the NVIDIA CNN architecture. 


####3. Creation of the Training Set & Training Process

I performed three laps of Track 1 while staying in the center of the road, so the model could learn good behaviour. I then performed roughly 40 recovery maneouvers from the sides of the road, to teach the car to recover to the center if it happens to go off course. 

I randomly flipped both the images and steering angles, to eliminate the left-turn bias gained from Track 1. 

