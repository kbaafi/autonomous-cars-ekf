#ifndef ekf_data_H_
#define ekf_data_H_


#include "Eigen/Dense"
#include <iostream>
#include <stdlib.h>


struct EKF_Data{
	  // state vector
  Eigen::VectorXd x_;

  // state covariance matrix
  Eigen::MatrixXd P_;

  // state transistion matrix
  Eigen::MatrixXd F_;

  // process covariance matrix
  Eigen::MatrixXd Q_;

  // measurement matrix
  Eigen::MatrixXd H_laser_;
	Eigen::MatrixXd H_radar_;

  // measurement covariance matrix
  Eigen::MatrixXd R_laser_;
	Eigen::MatrixXd R_radar_;
	
	float noise_ax_;
	float noise_ay_;

	// check whether the tracking toolbox was initialized or not (first measurement)
	bool is_initialized_;

	// previous timestamp
	long previous_timestamp_;
};

#endif
 
