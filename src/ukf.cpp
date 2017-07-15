#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {

  is_initialized_ = false;
  
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 30;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  n_x_ = 5;
  n_aug_ = 7;

  Xsig_pred_ = MatrixXd(n_x_, 2*n_aug_ + 1);

  weights_ = VectorXd(2*n_aug_ + 1);

  lambda_ = 3 - n_aug_;

  time_us_ = 0.1;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  if (!is_initialized_) {
    /**
    TODO:
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
    */
    // first measurement
    ukf.x_ = VectorXd(5);
    ukf.x_ << 1, 1, 1, 1, 1;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      ukf.x_ <<  measurement_pack.raw_measurements_[0]*cos(measurement_pack.raw_measurements_[1]), 
                  measurement_pack.raw_measurements_[0]*sin(measurement_pack.raw_measurements_[1]),
                  0,
                  0, 
                  0;
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      ukf.x_ <<  measurement_pack.raw_measurements_[0], 
                  measurement_pack.raw_measurements_[1], 
                  0,
                  0, 
                  0; 
    }

    previous_timestamp_ = measurement_pack.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  Prediction();

}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */

  // GENERATING AUGMENTED SIGMA POINTS ----------------------------------------
  //create augmented mean vector
  VectorXd x_aug = VectorXd(7);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(7, 7);

  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug, 2 * n_aug + 1);

  //create augmented mean state
  x_aug.head(5) = ukf.x_;
  
  //create augmented covariance matrix
  MatrixXd Q = MatrixXd(2,2);
  Q << std_a*std_a, 0, 0, std_yawdd*std_yawdd;

  P_aug.topLeftCorner(n_x,n_x) = P;
  P_aug.bottomRightCorner(2,2) = Q; 
 
  //create square root matrix
  MatrixXd A_aug = P_aug.llt().matrixL();
  
  //create augmented sigma points
  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug; ++i){
      Xsig_aug.col(i+1) = Xsig_aug.col(0) + sqrt(lambda + n_aug)*A_aug.col(i);
      Xsig_aug.col(i+n_aug+1) = Xsig_aug.col(0) - sqrt(lambda + n_aug)*A_aug.col(i);
  }

  // PREDICTION -----------------------------------------------------------------------
  VectorXd x = VectorXd(7);
  VectorXd v1 = VectorXd(5);
  VectorXd v2 = VectorXd(5);
    
  float dt2 = 0.5*delta_t*delta_t;
  
  for (int i = 0; i < 2*n_aug + 1; i++){
      x = Xsig_aug.col(i);

      if (std::abs(x(4)) <= 0.00001){
          v1 << x(2)*cos(x(3))*delta_t, 
                x(2)*sin(x(3))*delta_t,
                0,
                x(4)*delta_t,
                0;
      }
      else {
          v1 << (x(2)/x(4))*(sin(x(3) + x(4)*delta_t) - sin(x(3))),
                (x(2)/x(4))*(-cos(x(3) + x(4)*delta_t) + cos(x(3))),
                0,
                x(4)*delta_t,
                0;
      }

      v2 << dt2*cos(x(3))*x(5), 
            dt2*sin(x(3))*x(5), 
            delta_t*x(5), 
            dt2*x(6), 
            delta_t*x(6);
  
      ukf.Xsig_pred.col(i) = x.head(5) + v1 + v2;
  }

  // PREDICTING STATE AND COVARIANCE MATRICES ------------------------------------------
  ukf.weights_(0) = lambda_/(lambda_ + n_aug_);
  for (int i = 1; i < 2*n_aug_ + 1; i++){
      ukf.weights_(i) = 1/(2*(lambda_ + n_aug_));
  }

  //predicted state mean
  ukf.x_.fill(0.0);
  for (int i = 0; i < 2*n_aug_ + 1; i++) {
    ukf.x_ = ukf.x_ + ukf.weights_(i)*ukf.Xsig_pred_.col(i);
  }

  //predicted state covariance matrix
  ukf.P_.fill(0.0);
  for (int i = 0; i < 2*n_aug_ + 1; i++) {  

    // state difference
    VectorXd x_diff = ukf.Xsig_pred.col(i) - ukf.x_;
    //angle normalization
    while (x_diff(3) > M_PI) x_diff(3) -= 2.*M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2.*M_PI;

    ukf.P_ = ukf.P_ + ukf.weights_(i)*x_diff*x_diff.transpose();
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug + 1);
  int n_z = 3;

  //transform sigma points into measurement space
  for (int i = 0; i < 2*n_aug_ + 1; i++) {  //2n+1 simga points

    // extract values for better readibility
    double p_x = ukf.Xsig_pred_(0,i);
    double p_y = ukf.Xsig_pred_(1,i);
    double v  = ukf.Xsig_pred_(2,i);
    double yaw = ukf.Xsig_pred_(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
    Zsig(1,i) = atan2(p_y,p_x);                                 //phi
    Zsig(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
  }

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (int i=0; i < 2*n_aug+1; i++) {
      z_pred = z_pred + ukf.weights_(i) * Zsig.col(i);
  }

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug + 1; i++) {  //2n+1 simga points
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    S = S + ukf.weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z,n_z);
  R <<    std_radr*std_radr, 0, 0,
          0, std_radphi*std_radphi, 0,
          0, 0,std_radrd*std_radrd;
  S = S + R;

  ukf.x_ = z_pred;
  ukf.P_ = S;

}
