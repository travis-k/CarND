See other branches for previous projects
---

# Project 11 - Path Planning
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[//]: # (Image References)
[image1]: ./writeup_img/Capture.PNG "Success"

This is an outline of my submission for the path planning project. All files can be found in my CarND repository. Some of the files given with the project were reorganized into subfolders.
[https://github.com/travis-k/CarND.git](https://github.com/travis-k/CarND.git)

The Project
---

### Overview
In this project my goal was to safely navigate around a virtual highway with other traffic that is driving +-10 MPH of the 50 MPH speed limit. I was provided the car's localization and sensor fusion data, there is also a sparse map list of waypoints around the highway. The car tries to go as close as possible to the 50 MPH speed limit, which means passing slower traffic when possible, while other cars try to change lanes too. The car avoids hitting other cars at all cost as well as driving inside of the marked road lanes at all times, unless going from one lane to another. The car is able to make one complete loop around the 6946m highway. Since the car is trying to go 50 MPH, it takes a little over 5 minutes to complete 1 loop. Also the car does not experience total acceleration over 10 m/s^2 and jerk that is greater than 10 m/s^3.

### Description of the Model
My path planning follows the following logic:

1. The cars path is created using the spline.h library. The path is planned over the next 30 meters, using a number of points to smooth out the map coordinates. 

2. The car will slow down at a rate of 5 m/s^3 if it detects it is too close to the car in front. The car will accelerate at 5 m/s^3 if it detects no objects in front of it and it is below 49.5 mph. 

3. The car constantly monitors gaps in the neighbouring lanes. It assumes a lane change is safe, and at every timestep scans the neighbouring lanes in the vicinity of the car. If another car is detected, the car marks a lane change in that direction as unsafe.

4. The lane changes are executed using the spline generated at every step. If a lane change is desired, the Frenet "d" coordinates of the goal is shifted to the desired lane and the spline smooths out the transition to avoid jerkiness

### Results

Using the above system, the vehicle is capable of traveling around the track with no incidents. Below is an image where it did 4.43 miles in 7 minutes without incident. 

![Success of the Path Planning Model][image1]

### Future Work
Below are ideas I will implement to improve this path planner.

1. Currently, the car only looks to change lanes when it is actively decelerating due to a close car in front. This should be changed to open a bigger window for lane changing

2. The car has a boolean accelerate/decelerate to avoid colliding with the car in front. This should be switched to a PID or something more complex to match the speed of the car in front smoothly

3. The car does not take in to account the speeds of the vehicles in the other lane (or its own speed) when determining whether a lane change is safe. The safe gap setting is a fixed value. This should be changed to be more dynamic.

4. The car should return to the middle lane when done using the passing lane, if there is space and that lane is moving at the speed limit. 

