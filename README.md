Project 10 - MPC
---
The goals / steps of this project are the following:

* Implement an MPC for a car on a racetrack
* Ensure the car can travel around the track without hitting boundaries

**Description of the Model **
The controller was implemented as described throughout the lessons. The state is found. The constraints are set up in the solver with the provided lower and upper bounds. The simulation is run with the variables and constraints and their bounds, using the cost function. Different weights were implemented to the cost function to "tune" it to behave correctly. The solution is found which minimizes the cost function. 

**Reasoning in Determining N and dt**
The number of timestep size and timestep duration were tuned manually to allow the car to travel a lap of the track without leaving the road. With smaller or larger dt values, the car behaved erratically at times. A smaller dt provides the simulation with finer resolution, but requires larger N to achieve a good track fit to the path. With a large N, computation time increases. Increasing both N and dt allows the simulation to look farther down the road, but predicting more than a few seconds into the future is not necessarily important for the MPC.

**Preprocessing of Waypoints**
The waypoints are found using the global x and y location of the waypoints, and the angle psi. The distance to these waypoints from the vehicle is found, which effective centers the waypoints around the vehicle as the origin. These waypoints, with the vehicle at (0,0), are fitted with a third-order polynomial using polyfit, as was done in the lessons. 

**Latency Handling**
We run our simulation starting with the actuations from our previous timestep, which accounts for latency. This is in MPC.cpp, lines 102-105. This way, the results from the current simulation are fed back in to the next timesteps simulation. This is the same technique used in the lessons. 



