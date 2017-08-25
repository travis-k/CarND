Project 9 - PID Controller
---
The goals / steps of this project are the following:

* Implement a PID controller for a car on a racetrack
* Ensure the car can travel around the track without hitting boundaries

The PID controller was implemented, and the gains were tuned manually to allow the car to travel a lap of the track without leaving the road.

In the future, a twiddle algorithm will be used to train the model to minimize the cross track error. 

For now, the manual tuning consisted of getting the PD controller to safetly controll the car. The proportional gain was tuned so the car would return quickly
to center, and the differential gain raised to minimize the overshoot and oscillations. The integrated gain was kept quite low, as the car did not seem to have a bias in
this case. The car would return to center with just the PD controller.


