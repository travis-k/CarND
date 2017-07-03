Project 6 - Extended Kalman Filter
---
The goals / steps of this project are the following:

* Implement an extended Kalman filter in C++ to fuse lidar and radar inputs to track motion of a simulated vehicle in real-time

* Reduce mean squared error of the filter below the necessary threshold

* Identify the strengths and weaknesses of the lidar and radar data by isolating them and observing their error


Pipeline
---
* The Kalman filter follows the method described in the lessons

* Matrix H and Hj are stored separately in the EKF object, with Hj being updated before a radar estimation update

* Initial estimates use the first measurement location, and assign a speed of zero

* Corrections are added to avoid issues in dividing by zero

* Corrections are made to angles, ensuring they are normalized to -pi to pi

Results 
---
* The implemented Kalman filter for both datasets one and two are below the required thresholds. 

![alt text](/writeup_img/Dataset-1.PNG "Dataset 1")
![alt text](/writeup_img/Dataset-2.PNG "Dataset 2")

* Individually, the radar and lidar measurements result in a poor RMSE.

Lidar only:
![alt text](/writeup_img/lidar-only.PNG "Lidar Only")

Radar only:
![alt text](/writeup_img/radar-only.PNG "Radar Only") 