#ifndef ekf_H_
#define ekf_H_


#include "Eigen/Dense"
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include "ekf_data.h"
#include "ekf.h"
#include "measurement_package.h"
#include <vector>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace EKF{
	  /**
   * Init Initializes Kalman filter
   * @param x_in Initial state
   * @param P_in Initial state covariance
   * @param F_in Transition matrix
   * @param H_in Measurement matrix
   * @param R_in Measurement covariance matrix
   * @param Q_in Process covariance matrix
   */
  //void Init(Eigen::VectorXd &x_in, Eigen::MatrixXd &P_in, Eigen::MatrixXd &F_in,
  //    Eigen::MatrixXd &H_in, Eigen::MatrixXd &R_in, Eigen::MatrixXd &Q_in);
	void Init (EKF_Data &data,VectorXd x_in, MatrixXd P_in, MatrixXd F_in, MatrixXd H_laser_in, MatrixXd H_radar_in, MatrixXd R_laser_in, MatrixXd R_radar_in, MatrixXd Q_in );

	void Init(EKF_Data &data);

  /**
   * Prediction Predicts the state and the state covariance
   * using the process model
   * @param delta_T Time between k and k+1 in s
   */
  void Predict(EKF_Data &data);

  /**
   * Updates the state by using standard Kalman Filter equations
   * @param z The measurement at k+1
   */
  void Update(EKF_Data &data,const Eigen::VectorXd &z);

  /**
   * Updates the state by using Extended Kalman Filter equations
   * @param z The measurement at k+1
   */
  void UpdateEKF(EKF_Data &data,const Eigen::VectorXd &z);

	void ProcessMeasurement(EKF_Data &data,const MeasurementPackage &measurement_pack);
}

namespace Utils{
	Eigen::VectorXd CalculateRMSE(const vector<Eigen::VectorXd> &estimations,
                              const vector<Eigen::VectorXd> &ground_truth);
	Eigen::MatrixXd CalculateJacobian(const VectorXd& x_state);
	Eigen::VectorXd CalculateH_x(const Eigen::VectorXd& x_state);
}


#endif
