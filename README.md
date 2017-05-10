See other branches for previous projects
---

[//]: # (Image References)
[image1]: ./writeup_images/bboxes_and_heat.png "Boxes and Heat Plot"
[image2]: ./writeup_images/car_not_car.png "Example of Training Data"
[image3]: ./writeup_images/HOG_example.jpg "Example of HOG features"
[image4]: ./writeup_images/labels.png "Example of the Labels from the Heat Data"
[image5]: ./writeup_images/search.png "Searching for Cars"
[image6]: ./writeup_images/pipelineoutput.jpg "Pipeline Output"

This is an outline of my submission for Project 5: Vehicle Detection and Tracking. All files can be found in my CarND repository. Some of the files given with the project were reorganized into subfolders.
[https://github.com/travis-k/CarND.git](https://github.com/travis-k/CarND.git)

The Project
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.


Extracting HOG Features from Training Images
---

This project used training data from the GTI vehicle image database, and the KITTI vision benchmark suite. All images were 64x64 resolution, and kept in a folder labelled training_data (with vehicle/ and non-vehicle/ subfolders). 

The data preprocessing (images to feature vectors) was done in data_processing.py

Examples of raw images are shown here.

![Example of Training Data][image2]

This data was read in and converted to YCrCb colour space, and a feature vector was created for each image. This vector consisted of binned colour features, a colour histogram, and the HOG features on all three channels.

![Example of HOG features][image3]

The values for the HOG features are: orientation = 9, pixels_per_cell = 8, cell_per_block = 2. These can be seen in lines 21-22 of data_processing.py. The reason these values were kept is because they worked well in both the lessons and project.

The feature vector is normalized in lines 25-27 of data_processing.py

The feature vector, classifications, and X_scaler are saved in a pickle filed named train_data.p. This is so the preprocessing only needs to be done once.

Training the Classifier
---

The linear SVC classifier is trained in p5_pipeline.py beginning at line 102. The training only occurs of svc_model.p does not exist in the local folder, otherwise the pretrained model is loaded instead. This is done so the model doesn't have to be trained every time the pipeline is run.

Accuracy of 98.6% was achieved with a 80/20 training/validation split using shuffled data.

Sliding Window Search
---

The function find_cars in helper_functions.py is used for the sliding window search. The function is used three times, in p5_pipeline.py lines 148-178. The first time is to focus on small cars near the horizon, so a scale of 1 is used and the boxes are restricted to the area around the horizon closer to the middle of the frame. The second call to find_cars is focused on medium-range, so the boxes are set to scale = 2 and focused a little lower in the frame. Lastly, close-range searching is done with larger boxes (scale = 2.5) and in the bottom 1/4 of the image.

The image or video frame is read in and converted to YCrCb in find_cars. The HOG is then performed on all channels and subdivided into appropriate cells, of the same size used in the processing of the training images (all values are the same as before). Cells_per_step was set to 1 which yielded better results. 

![Searching for Cars][image5]

A heat plot is used to help eliminate false positives and multiple detections on the same vehicle. This is done in p5_pipeline.py line 193-222. For still images, a threshold of 2 hits is needed to identify the area as a vehicle. An example of a heat plot using this criteria is shown here. 

![Boxes and Heat Plot][image1]

This is then used to create label data which can identify each individual car and put a single bounding box around that car.

![Example of the Labels from the Heat Data][image4]


Video Implementation
---
The pipeline was implemented on every frame of the video. For videos, an averaging of the last 10 frames is used in the heat plot to help avoid false positives and noise. This average is then used to create the bounding box around the cars. 


Discussion
---
The pipeline works fairly well for the given video. There are a couple of minor false positives, and the bounding boxes around the cars are not always the correct size.

This could be improved by augmenting or adding to the training data, as well as adding in more search window sizes and locations, instead of the 3 sizes currently used.

The program could also benefit from a more elaborate averaging between video frames, to further smooth out the detection and bounding box location and sizing.  