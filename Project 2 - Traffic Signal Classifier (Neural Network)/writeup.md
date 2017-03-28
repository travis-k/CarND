#**Traffic Sign Recognition** 

##Writeup

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./outside_images/web_imgs.png "Eight signs taken from the web."
[image2]: ./outside_images/classes.png "Examples of the 43 classifications."
[image3]: ./outside_images/dist.png "Distribution of classifications among training set."
[image4]: ./outside_images/13.png "Yield sign top 5 softmaxes"
[image5]: ./outside_images/14.png "Stop sign top 5 softmaxes"
[image6]: ./outside_images/18.png "General caution sign top 5 softmaxes"
[image7]: ./outside_images/24.png "Road narrowing sign top 5 softmaxes"
[image8]: ./outside_images/27.png "Pedestrian sign top 5 softmaxes"
[image9]: ./outside_images/28.png "Children crossing sign top 5 softmaxes"
[image10]: ./outside_images/33.png "Turn right ahead sign top 5 softmaxes"
[image11]: ./outside_images/36.png "Go straight or right sign top 5 softmaxes"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

##Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The data for this project was provided by Udacity. I imported the pickle files, which were already subdivided into training, validating and testing sets. I shuffle the training set to ensure no issues arise from the order. Then, I make sure the lengths are the same. These things are dealt with in code cells 1 and 2 of the IPYNB.

* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 42

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in code cell 3 of the IPYNB.  

It first displays an example of each classification from the training set.

![alt text][image2]

It then shows the distribution of these classifications among the training set, showing how some classes are more represented than others. 

![alt text][image3]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

In code cell 4, I normalize the RGB values about the origin, and scale them onto the range (-1,1). This is to increase the effectiveness of the optimizer. I also convert to grayscale, using [0.289 0.587 0.114] colour balance, which is standard weighting (apparently).

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

I used the data provided by Udacity, which is already subdivided into training, validation and testing. The division of the data can be seen in the above summary, or in the summary in the IPYNB code cell 2 output.

Time constraints limited me to using only the provided data, but I understand how augmenting the training data with new images and altered versions of the original images (different color, rotations, etc) would help to train the network much better.

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The model architecture is found in code cell 5 of the IPYNB.

My "inception" neural network is based off of the inception modules explained in the Udacity lessons on deep learning. It is also based on the archiceture outlined in this article (https://hacktilldawn.com/2016/09/25/inception-modules-explained-and-implemented/), albiet with some changes. 

The architecture features two inception modules, followed by two fully-connected layers. When training, the first fully-connected layer has dropout. The number of feature maps and the depths of the layers were reduced significantly, as their sizes were restricted by memory issues on the training computer.

The first inception module is as follows, with the input as 32x32x1 and the output concatenation 24 32x32 feature maps:    

* Input -> 1x1 convolution (6 32x32 feature maps) -> concatenation
* Input -> 1x1 convolution (3 32x32 feature maps) -> 3x3 convolution (6 32x32 feature maps) -> concatenation
* Input -> 1x1 convolution (3 32x32 feature maps) -> 5x5 convolution (6 32x32 feature maps) -> concatenation
* Input -> 3x3 max pooling (1 32x32 feature map) -> 1x1 convolution (6 32x32 feature maps) -> concatenation

The second is similar to the first with the input 24 32x32 feature maps from the previous module, and the output a flattened concatenation of 48 32x32 feature maps, resulting in a shape of 1x49152:
* Input -> 1x1 convolution (12 32x32 feature maps) -> flattened concatenation
* Input -> 1x1 convolution (3 32x32 feature maps) -> 3x3 convolution (12 32x32 feature maps) -> flattened concatenation
* Input -> 1x1 convolution (3 32x32 feature maps) -> 5x5 convolution (12 32x32 feature maps) -> flattened concatenation
* Input -> 3x3 max pooling (24 32x32 feature maps) -> 1x1 convolution (12 32x32 feature maps) -> flattened concatenation

Then the first fully connected layer, with the input the 1x49152 concatenation from the previous layer, and the output 1x128
* Input -> ReLU (with dropout for training)

Then the final layer, with a 1x128 input and a 1x43 output,
* Input -> output (logits)
 
####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Code cell 6 contains the setup for training the model, including assigning hyperparameters. I use a learning rate of 0.001, as anecdotally it provided the best final results in the quickest time. Further investigation would need to be done to improve this. The batch size was set to 128 due to memory constraints. The dropout for training was set at 0.5, and 1 for validation and testing. 

The number of epochs was chosen as 15, as that is typically where training accuracy reaches 100% and validation accuracy levels off (more or less).

Code cell 7 contains the actually training and validation regime, and follows similarly to the LeNet Udacity lab, with the addition of training accuracy outputs.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The training and validation accuracies are output with the training regime in code cell 7. The testing accuracy is printed out with code cell 8. 

My final model results were:
* training set accuracy of 100%
* validation set accuracy of ~95%
* test set accuracy of 93.5%

##What was the first architecture that was tried and why was it chosen?
I tried the LeNet architecture first (you can see it in the /misc/ folder). At first, with the vanilla LeNet, I could get ~89% validation accuracy. Using normalization and grayscale, and playing with the learning rate, I could get it up to ~94%.

##What were some problems with the initial architecture?
I believed I could get higher accuracy with the inception modules. 

##How was the architecture adjusted and why was it adjusted? 
I totally revamped it to the inception module oriented architecture (albiet a much smaller network than it should be, due to computational restrictions)

##Which parameters were tuned? How were they adjusted and why?
Batch size, learning rate, dropout and the layer sizes were tuned. Most of these were due to memory constraints, and the dropout was set high to try to avoid overfitting.

##What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

A higher dropout forces the network to adapt to missing inputs and create redundancies. I used 50% dropout, because I was unable to augment the training set with more data. I was drawn to the inception module, as it was recommended in the lessons for image recognition.

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

As stated above, I was drawn to the inception module after learning about it in the lessons. It includes a broad range of convolutions, so it lends itself well to image recognition.

My final training accuracy reached 100%, with my validation accuracy floating from 94.5-97%. I believe this means my model was overfitting the training data. I would have liked to have augmented the data to avoid this. My score of 93.5% on the test set is, I believe, acceptable, but again probably means my model was overfitting. 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image1]

The first image might be difficult to classify because of the detailed figures in the center of the sign. As well, the first three images, as well as images six and eight, all have the same shape and border colour.

Images four and five also share the same border colour, and similar drawings in the center.

Image seven, the stop sign, may be difficult to identify due to the comparatively large amount of text on this low resolution image.  

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Children crossing		| Children crossing   							| 
| General caution    	| General caution 								|
| Pedestrians 			| Right of way at next intersection				|
| Stay straight or right| Stay straight or right					 	|
| Right turn only		| Right turn only     							|
| Right lane ends		| Right lane ends     							|
| Stop		  			| Road works	      							|
| Yield   				| Yield       									|


The model was able to correctly guess 6 out of 8 traffic signs, which gives an accuracy of 75%. This is less than the testing accuracy using the Udacity pickle data. I believe this discrepancy is due to the sign images I chose, which are vector image representations of the signs rather than actual photographs of the signs. It is also likely a result of poor training, with more data augmentation needed in the training set and wider layers in network.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook, and the percentages are shown there. 

The yield sign was categorized succesfully by my neural network with a 100% softmax.
![alt text][image4]

The stop sign exposed the biggest flaws in my neural network, as it categorized it as a road work sign with 98% accuracy. Classification 14 did not even appear in the top 5 softmaxes. This probably means that I need to augment the training data with much (much much) more, and let the network train longer than 15 epochs. 
![alt text][image5]

The general caution sign was categorized succesfully by my neural network with a 100% softmax.
![alt text][image6]

The road narrowing sign was categorized succesfully by my neural network with a 99.97% softmax.
![alt text][image7]

The pedestrian crossing sign was wrongly categorized by my neural network with a 99.99% softmax. It thought it was a "right of way at next crossing" sign, which actually shares many graphical similarities to the pedestrian crossing sign. Again, this is likely due to a poor training regime with insufficient training data.
![alt text][image8]

The children crossing sign was categorized succesfully by my neural network with a 99.99% softmax, which was interesting. It is a very detailed sign, but still categorized correctly whereas the pedestrian sign was not.
![alt text][image9]

The turn right ahead sign was categorized succesfully by my neural network with a 99.99% softmax.
![alt text][image10]

The keep straight or turn right sign was categorized succesfully by my neural network with a 53.6% softmax, with road work coming in second place with 46.37%. Again, likely an idication that my training regime was poor.
![alt text][image11]
