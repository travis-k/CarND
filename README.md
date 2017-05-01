# Writeup: Finding Lane Lines on the Road

This is an outline of my submission for Project 1: Finding Lane Lines. All files can be found in my CarND repository. Some of the files given with the project were reorganized into subfolders.

[https://github.com/travis-k/CarND.git](https://github.com/travis-k/CarND.git)

My project will output all test images and videos with both raw Hough lines and averaged lane lines, in separate folders. 

[//]: # (Image References)
[image1]: ./setting_optimization/desired_images/solidWhiteRight.jpg "Desired Hough Lines"
[image2]: ./setting_optimization/test_images_output/solidWhiteCurve.jpg "Output using Optimizer Settings"

## Reflection

### Pipeline description

The pipeline for my project follows closely to the Finding Lane Lines lesson on Udacity. The pipeline is as follows:

1. **Import the image or video, with videos being a sequence of images.**
2. **Convert the image to grayscale.**
3. **Use Gaussian blur to blur the image.**
4. **Apply Canny edge detection.**
5. **Mask the image, so the focus is on the triangular region where the lane lines will be.**
6. **Use the Hough transformation to identify lines.**
7. **Separate the Hough lines into left and right lane lines, and average them individually to arrive at the left and right lines.**
8. **Plot these left and right lines onto the image.**

The draw_lines function is where my project has been more heavily modified. To identify the left and right lane lines individually, I find the slope of all of the Hough lines. To do this, I take the weighted average of the slopes on each side, where the weight is the line length. The reason I do this is because the longer lines are more likely to be in the correct direction, as opposed to other small lines which weren't intended to be picked up at all. In order to determine which lines contribute to the left or right lanes, the program assumes the lines with a positive slope are likely to be on the left, as they go upwards and to the right. The lines with negative slope are likely to be on the right, as they go upward and to the left. I ignore lines which have positive slope on the right side of the image, and lines with negative slope on the left side of the image, as these are most likely noise. 

Next, I find the average of all points on the left and right lane lines. Then, using the average slope for each side, I extend the line to the bottom of the image and upwards closer to the horizon. This provides me with the averaged left and right lane lines, which I then draw on the image. 

### Potential shortcomings with my current pipeline

It is unlikely that my lane detection program would work with video taken at night, or in weather such as rain or snow. It has trouble in corners and with shadows, as seen in the results of my "challenge" video. It also would likely not work when a car is in the mask area. It also sometimes picks up reflections in the windshield. 

I tried to keep the mask area as a ratio of the image size, but it would not work with a different camera angle.

### Possible improvements to my pipeline

The settings with which I perform the Gaussian blur, Canny edge detection, and Hough line detection are rather arbitrarily chosen. I did make an attempt to determine the correct settings by creating an image of the Hough lines that I wanted (by doing it manually), and backdriving my function with an optimizer to identify the settings that would provide me with an image similar to what I wanted. This attempt can be seen in the setting_optimization folder.

It was unfortunately unsuccessful in the attempts I made early on. The settings returned by the optimizer sometimes ignored a left or right lane altogether, so it favoured very strict settings for the Hough transformation. I would like to return to this in the future to see if I could improve my results. It could be more easily done with a multi-objective optimization, but that is not readily available in Python (I come from a MATLAB background).

This is the desired output of the Hough lines, created manually:
![Desired Hough Lines][image1]

This is the Hough line output using the results from the optimizer:
![Output using Optimizer Settings][image2]