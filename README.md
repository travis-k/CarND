See other branches for previous projects
---

[//]: # (Image References)
[image1]: ./writeup_images/calibration.png "Calibration"
[image2]: ./writeup_images/colourbinary.png "Colour Binary"
[image3]: ./writeup_images/detectinglines.png "Detecting Lane Lines"
[image4]: ./writeup_images/distortion.png "Correcting Distortion"
[image5]: ./writeup_images/masking.png "Masking the Image"
[image6]: ./writeup_images/pipelineoutput.png "Pipeline Output"
[image7]: ./writeup_images/warping.png "Changing the Perspective"

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