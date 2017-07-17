Project 7 - Unscented Kalman Filter
---
The goals / steps of this project are the following:

* Implement an unscented Kalman filter in C++ to fuse lidar and radar inputs to track motion of a simulated vehicle in real-time
* Reduce mean squared error of the filter below the necessary threshold
* Compare the results of this filter with the EKF from the previous project


Pipeline
---
* The UKF follows the method described in the lessons
* Initial estimates use the first measurement location, and assign a speed of zero as with the EKF project
* Corrections are added to avoid issues in dividing by zero
* Corrections are made to angles, ensuring they are normalized to -pi to pi
* The parameters were tuned manually by observing their effects on RMSE
* The NIS method will later be used to improve the set parameters

Results 
---
* The implemented unscented Kalman filter for both datasets one and two are shown below. For the first dataset, the errors are significantly lower for the x-component of the velocity.
* The RMSE values for the UKF are lower than those of the EKF across the board, for the same datasets
* This change in RMSE is likely due to the fact that the car is constantly turning, a situation in which the EKF has issues due to its linearized rate of change estimation  

![alt text](/writeup_img/Dataset-1.PNG "Dataset 1")
![alt text](/writeup_img/Dataset-2.PNG "Dataset 2")
