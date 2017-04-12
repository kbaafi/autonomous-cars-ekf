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
using std::vector;

void EKF::Init (EKF_Data &data,VectorXd x_in, MatrixXd P_in, MatrixXd F_in, MatrixXd H_laser_in, MatrixXd H_radar_in, MatrixXd R_laser_in, MatrixXd R_radar_in, MatrixXd Q_in ){

	data.x_ = x_in;
	data.P_ = P_in;
	data.F_ = F_in;
	data.Q_ = Q_in;
	data.H_laser_ = H_laser_in;
	data.H_radar_ = H_radar_in;
	data.R_laser_ = R_laser_in;
	data.R_radar_ = R_radar_in;

}

void EKF::Init (EKF_Data &data){
	// initializing matrices
	MatrixXd R_laser = MatrixXd(2, 2);
	MatrixXd R_radar = MatrixXd(3, 3);
	MatrixXd H_laser = MatrixXd(2, 4);
	MatrixXd P = MatrixXd(4, 4);
	MatrixXd F = MatrixXd(4, 4);
	MatrixXd Q = MatrixXd(4, 4);
	MatrixXd H_radar = MatrixXd(3, 4);

	//measurement covariance matrix - laser
	R_laser << 0.0225, 0,
							0, 0.0225;

	//measurement covariance matrix - radar
	R_radar << 0.09, 0, 0,
							0, 0.0009, 0,
							0, 0, 0.09;
	
	H_laser << 1,0,0,0,
		    			0,1,0,0;

	// this will be updated later in the measurement update stage when we receive each measurement
	// setting to zeros is fine
	H_radar << 0,0,0,0,
							0,0,0,0,
							0,0,0,0;

	Q << 0,0,0,0,
				0,0,0,0,
				0,0,0,0,
				0,0,0,0;

	P << 1, 0, 0, 0,
			  0, 1, 0, 0,
			  0, 0, 1000, 0,
			  0, 0, 0, 1000;

	F << 1, 0, 1, 0,
			  0, 1, 0, 1,
			  0, 0, 1, 0,
			  0, 0, 0, 1;


	VectorXd x = VectorXd(4);
	x << 1,1,1,1;

	data.noise_ax_ = 9;
	data.noise_ay_ = 9;

	data.x_ = x;
	data.P_ = P;
	data.F_ = F;
	data.Q_ = Q;
	data.H_laser_ = H_laser;
	data.H_radar_ = H_radar;
	data.R_laser_ = R_laser;
	data.R_radar_ = R_radar;
	data.is_initialized_ = false;
	data.previous_timestamp_ = 0;
	
}


void EKF::Predict(EKF_Data &data){
	data.x_ = data.F_ * data.x_;
	MatrixXd Ft = data.F_.transpose();
	data.P_ = data.F_ * data.P_ * Ft + data.Q_;
}


void EKF::Update(EKF_Data &data,const Eigen::VectorXd &z){
	VectorXd z_pred = data.H_laser_ * data.x_;
	VectorXd y = z - z_pred;
	MatrixXd Ht = data.H_laser_.transpose();
	MatrixXd S = data.H_laser_ * data.P_ * Ht + data.R_laser_;
	MatrixXd Si = S.inverse();
	MatrixXd PHt = data.P_ * Ht;
	MatrixXd K = PHt * Si;

	//new estimate
	data.x_ = data.x_ + (K * y);
	long x_size = data.x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	data.P_ = (I - K * data.H_laser_) * data.P_;
}


void EKF::UpdateEKF(EKF_Data &data,const Eigen::VectorXd &z){
	Eigen::VectorXd h_x = Eigen::VectorXd(3);

	// calculate h_x
	h_x = Utils::CalculateH_x(data.x_);

	//update the jacobian
	data.H_radar_ = Utils::CalculateJacobian(data.x_);

	VectorXd z_pred = h_x;
	VectorXd y = z - z_pred;

	MatrixXd Ht = data.H_radar_.transpose();
	MatrixXd S = data.H_radar_ * data.P_ * Ht + data.R_radar_;
	MatrixXd Si = S.inverse();
	MatrixXd PHt = data.P_ * Ht;
	MatrixXd K = PHt * Si;

	//new estimate
	data.x_ = data.x_ + (K * y);
	long x_size = data.x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	data.P_ = (I - K * data.H_radar_) * data.P_;
	
}

void EKF::ProcessMeasurement(EKF_Data &data,const MeasurementPackage &measurement_pack){
	/*****************************************************************************
	*  Initialization
	****************************************************************************/

  if (!data.is_initialized_) {
	/**
	TODO:
	* Initialize the state ekf_.x_ with the first measurement.
	* Create the covariance matrix.
	* Remember: you'll need to convert radar from polar to cartesian coordinates.
	*/
	// first measurement
	cout << "EKF: " << endl;
	//ekf_.x_ = VectorXd(4);
	//ekf_.x_ << 1, 1, 1, 1;

		if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {

			// Initialize Radar state
			/**
			Convert radar from polar to cartesian coordinates and initialize state.
			*/
			float rho = measurement_pack.raw_measurements_(0);
			float phi = measurement_pack.raw_measurements_(1);

			data.x_[0] = rho*cos(phi);
			data.x_[1] = rho*sin(phi);
		}
		else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
			/**
			Initialize Lidar state.
			*/
			data.x_[0] = measurement_pack.raw_measurements_(0);
			data.x_[1] = measurement_pack.raw_measurements_(1);
		}

		data.previous_timestamp_ = measurement_pack.timestamp_;
		data.is_initialized_ = true;
		// done initializing, no need to predict or update
	
		return;
	}

	/*****************************************************************************
	*  Prediction
	****************************************************************************/

	/**
	TODO:
	* Update the state transition matrix F according to the new elapsed time.
	- Time is measured in seconds.
	* Update the process noise covariance matrix.
	* Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
	*/

	// dt - expressed in seconds
	float dt = (measurement_pack.timestamp_ - data.previous_timestamp_) / 1000000.0;
	data.previous_timestamp_ = measurement_pack.timestamp_;

	float dt_2 = dt * dt;
	float dt_3 = dt_2 * dt;
	float dt_4 = dt_3 * dt;
		

	// Setting up transition matrix F
	data.F_(0, 2) = dt;
	data.F_(1, 3) = dt;

	// Setting up process noise matrix Q
	data.Q_ = MatrixXd(4, 4);
	data.Q_ <<  dt_4/4*data.noise_ax_, 0, dt_3/2*data.noise_ax_, 0,
			   			0, dt_4/4*data.noise_ay_, 0, dt_3/2*data.noise_ay_,
			   			dt_3/2*data.noise_ax_, 0, dt_2*data.noise_ax_, 0,
			   			0, dt_3/2*data.noise_ay_, 0, dt_2*data.noise_ay_;
	
	

	EKF::Predict(data);

	/*****************************************************************************
	*  Update
	****************************************************************************/

	/**
	TODO:
	* Use the sensor type to perform the update step.
	* Update the state and covariance matrices.
	*/

	if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
		// Radar updates

		EKF::UpdateEKF(data,measurement_pack.raw_measurements_);
	} else {
		// Laser updates
		EKF::Update(data,measurement_pack.raw_measurements_);
	}

	// print the output
	cout << "x_ = " << data.x_ << endl;
	cout << "P_ = " << data.P_ << endl;
}

Eigen::VectorXd Utils::CalculateRMSE(const vector<Eigen::VectorXd> &estimations,
                              const vector<Eigen::VectorXd> &ground_truth) {
  /**
  TODO:
    * Calculate the RMSE here.
  */

	VectorXd rmse(4);
	rmse << 0,0,0,0;

    // TODO: YOUR CODE HERE

	// check the validity of the following inputs:
	//  * the estimation vector size should not be zero
	//  * the estimation vector size should equal ground truth vector size
	// ... your code here
	if(estimations.size()==0||estimations.size()!=ground_truth.size()){
	    cout<<"Error: Estimations size is zero or not equal to ground truth size";
	    return rmse;
	}
	
	VectorXd residual;
  VectorXd residual_sq;
	//accumulate squared residuals
	for(int i=0; i < estimations.size(); ++i){
        // ... your code here
        VectorXd residual = estimations[i]-ground_truth[i];
        VectorXd residual_sq = residual.array()*residual.array();
        rmse+=residual_sq;
	}

	//calculate the mean
	// ... your code here
	rmse = rmse/estimations.size();

	//calculate the squared root
	// ... your code here
	rmse = rmse.array().sqrt();

	//return the result
	return rmse;
}

Eigen::MatrixXd Utils::CalculateJacobian(const VectorXd& x_state) {
	
	MatrixXd Hj(3,4);
	//recover state parameters
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);

	float sum_sq = px*px+py*py;
	float sqrt_sum_sq = sqrt(sum_sq);
	float sum_sq_3_2 = sum_sq*sqrt_sum_sq;

	if (fabs(sum_sq)<0.0001){
		cout << "CalculateJacobian () - Error - Division by Zero" << endl;
		return Hj;
	}

	float drho_dpx = px/sqrt_sum_sq;
	float drho_dpy = py/sqrt_sum_sq;
	float dphi_dpx = -1.0*py/sum_sq;
	float dphi_dpy = px/sum_sq;
	float drhodot_dpx = py*(vx*py-vy*px)/sum_sq_3_2;
	float drhodot_dpy = px*(vy*px-vx*py)/sum_sq_3_2;

	Hj << drho_dpx,drho_dpy,0,0,
				dphi_dpx,dphi_dpy,0,0,
				drhodot_dpx,drhodot_dpy,drho_dpx,drho_dpy;

	return Hj;
	
}

Eigen::VectorXd Utils::CalculateH_x(const Eigen::VectorXd& x_state){
	VectorXd h_x(3);
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);

	float sum_sq = px*px+py*py;
	float sqrt_sum_sq = sqrt(sum_sq);

	if (fabs(sum_sq)<0.0001){
		cout << "CalculateH_x () - Error - Division by Zero" << endl;
		return h_x;
	}

	float rho = sqrt_sum_sq;
	float phi = atan2(py,px);		//normalized phi
	float rho_dot = (px*vx+py*vy)/sqrt_sum_sq;
	
	
	h_x<<rho,phi,rho_dot;
	return h_x;
}
