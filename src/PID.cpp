#include "PID.h"
#include <iostream>

using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd) {
    PID::Kp = Kp;
    PID::Ki = Ki;
    PID::Kd = Kd;
}

void PID::UpdateError(double cte) {
    // updateCount++;

    d_error = cte - p_error;
    p_error = cte;
    i_error += cte;
    
    // do twiddle

    // if (step == 0) {
    //     p[currIdx] += dp[currIdx];
    // step++;
    // } else if (step == 1) {
    //     if (error < best_error) {
    //         best_error = error;
    //         dp[currIdx] *= 1.1;
    //     currIdx++;
    //         step = 0;
    //     } else {
    //         p[currIdx] -= 2 * dp[currIdx];
    //         step++; 
    // }
    // } else {
    // if (error < best_error) {
    //     best_error = error; 
    //     dp[currIdx] *= 1.1;
    // } else {
    //     p[currIdx] += dp[currIdx];
    //     dp[currIdx] *= 0.9;
    // }
    //     step = 0;
    //     currIdx = (currIdx + 1) % 3;
    // }
}

double PID::TotalError() {
   return -Kp * p_error - Kd * d_error - Ki * i_error;
}
