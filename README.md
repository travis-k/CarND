See other branches for previous projects
---

# Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[//]: # (Image References)
[image1]: ./writeup_images/calibration.png "Calibration"
[image2]: ./writeup_images/colourbinary.png "Colour Binary"
[image3]: ./writeup_images/detectinglines.png "Detecting Lane Lines"
[image4]: ./writeup_images/distortion.png "Correcting Distortion"
[image5]: ./writeup_images/masking.png "Masking the Image"
[image6]: ./writeup_images/pipelineoutput.png "Pipeline Output"
[image7]: ./writeup_images/warping.png "Changing the Perspective"

This is an outline of my submission for Project 4: Advanced Lane Detection. All files can be found in my CarND repository. Some of the files given with the project were reorganized into subfolders.
[https://github.com/travis-k/CarND.git](https://github.com/travis-k/CarND.git)

The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

Camera Calibration
---

To correct for camera distortion, the various 9x6 chess board images in the 'camera_cal/' folder were used to calculate the distortion correction values. This was done using OpenCV, using built-in functions. 

The distortion correction values are stored in a local file, 'camera_calibration_saved.p'. If this file isn't present when the program is run, it will generate it using the calibration images in the "camera_cal/" folder. 

![Calibration][image1]

Distortion Correction
---

The first thing done after reading in the distortion correction values is using them to undistort the image from the car's camera. This is done using OpenCV and the values found in the above section. 

![Distortion Correction][image4]

Color Transformations and Gradients
---

With the undistorted camera image, the image is converted to HLS image space. The Sobel in the x direction is calculated on the light channel. This is thresholded with values [20,100] to form a colour binary. The saturation channel alone is thresholded to form a second colour binary. These two are then combined with an OR operator to form the final colour binary.

The reason the S and L channels are used is because they provided the best experimental results. More experimentation should be done on this in the future to improve the results on the challenge videos. 

The combined colour binary is shown below. 
![Colour Binary][image2]

The colour binary is then masked, using the same area used later on to perform the perspective transformation.

![Colour Binary Masking][image5]

Perspective Transformation
---

The source area for the perspective transformation was found by casting a trapezoid which followed the lane lines from the bottom of the image to just below the horizon on one of the straight test images. This was done so the sides of the perspective transformation follow the road lines. Then, padding was added to the sides of the trapezoid so it was evenly spaced on the outside of the lane lines. This area was used for the perspective transformation source matrix.

The destination matrix was a rectangle of much greater height. For the straight road test images, it was ensured that the perspective transformation resulted in straight lines in the warped image.

![Perspective Transformation][image7]

Identifying Lane Lines
---
The lane lines are found using the histogram/sliding window method. A histogram is performed on the lower quarter of the image to find the base of the lane lines. The left line is found on the left half of the image, and the right is found on the right. This is to avoid the program from detecting other lines and, for example, deciding both lines are on the left side of the car.

Once the base of the lines are found, 20 sliding windows are used to track the lines upwards. Polynomials are then fit to these windows, forming the equations of the left and right lane lines.

![Identifying Lane Lines][image3]

Radius of Curvature and Offset Calculation
---

The radius of curvature calculation was found using the provided radius of curvature equation. The curvature of the corner in the project video was found to be approximately 1000m, which is the same order of magnitude as described by Udacity. This is obviously an approximation, as it depends heavily on the values used in the perspective transformation.

The curvature goes to high numbers on the straight section of the road, as expected.

The offset from the center of the lane was determined by averaging the base of the left and right lane lines and taking the difference from the center of the image, which is at 640 pixels. This was then converted from pixels to meters as described by Udacity. This value hovers around zero, but it also assumes that the camera is mounted dead center in the car.

Warping Back to Original Perspective
---

Once the lane lines are calculated, the area inside the detected lane is masked off. This shape is then transformed back into the original perspective, using the inverse warp transformation.

![Calibration][image6]

Pipeline Adjustment for Videos
---
When using the program for videos, a smarter search is performed to find the lane lines if they were detected in the last frame. The program uses the previous location of the lane lines, and searches around those lines for the new lines. 

The program also uses the moving average of the past 15 frames in determining the lane line equations. This is done by averaging the coefficients over the last 15 frames. This smooths out the lane detection in the video, and helps to eliminate outliers. 

If the lane lines become to close together, or both end up on one half of the screen, they are recalculated from scratch.

Issues or Difficulties in This Project
---

My program works well for test images and the test video. However, it does quite poorly on the challenge videos. 

On "challege_video.mp4", my program typically estimates the lane to be wider than it is. It picks up shadows on the left barrier, rather than the yellow line.  

On "harder_challenge_video.mp4", my program does poorly with sharp turns. It also has trouble distinguishing the lane line from the guard rail.

These challenge video results could be improved by further tweaking the parameters for colour transformation, gradient calculations, masking, warping, and line detection.
