#include "PID.h"
#include <iostream>
#include <numeric>

/**
 * TODO: Complete the PID class. You may add any additional desired functions.
 */

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp_, double Ki_, double Kd_) {
  /**
   * TODO: Initialize PID coefficients (and errors, if needed)
   */
  this->Kp = Kp_;
  this->Ki = Ki_;
  this->Kd = Kd_;

  this->p_error = 0.0;
  this->i_error = 0.0;
  this->d_error = 0.0;

  //twiddle delta coefficents
  this->d_Kp = 1;
  this->d_Kd = 1;
  this->d_Ki = 1;
  this->best_err = 10000;

  // Previous cte.
  this->prev_cte = 0.0;

}

void PID::UpdateError(double cte) {
  /**
   * TODO: Update PID errors based on cte.
   */
  // Proportional error.
  this->p_error = cte;

  // Integral error.
  this->i_error += cte;

  // Diferential error.
  this->d_error = cte - prev_cte;
  this->prev_cte = cte;
}

double PID::TotalError() {
  /**
   * TODO: Calculate and return the total error
   */
  double steer = -1* ((this->p_error * this->Kp) + 
         (this->i_error * this->Ki) + 
         (this->d_error * this->Kd));  // TODO: Add your total error calc here!
  if (steer < -1) {
    steer = -1;
  }
  if (steer > 1) {
    steer = 1;
  }
  return steer;
}

int arraySum(double a[], int n)  
{ 
    double initial_sum  = 0;  
    return std::accumulate(a, a+n, initial_sum); 
} 


void PID::Twiddle(double err) {

  //Choose an initialization parameter vector
  double p[3] = {this->Kp, this->Kd, this->Ki};
  //Define potential changes
  double dp[3] = {this->d_Kp, this->d_Kd, this->d_Ki};

  double threshold = 0.001;

  std::cout<<"arraySum(dp , 3) ="<< arraySum(dp , 3) << std::endl;
  if (arraySum(dp , 3) > threshold)
  {
    for (int i = 0; i < 3; i++ ) {
          p[i] += dp[i];

          if(err < best_err){  //There was some improvement
              best_err = err;
              dp[i] *= 1.05;
          } else {  // There was no improvement
              p[i] -= 2*dp[i];  //Go into the other direction

              if (err < best_err){  //There was an improvement
                  best_err = err;
                  dp[i] *= 1.05;
              }else{  // There was no improvement
                  p[i] += dp[i];
                  //As there was no improvement, the step size in either
                  //direction, the step size might simply be too big.
                  dp[i] *= 0.95;
              }
            }
      }
      //save parameters
      this->Kp = p[0];
      this->Kd = p[1];
      this->Ki = p[2];
      std::cout<<"Twiddle-Hyperparameter =["<< this->Kp << " ; " << this->Kd <<" ; "<< this->Ki << "]"<< std::endl;
      std::cout<<"Twiddle-d_parameter =["<< this->d_Kp << " ; " << this->d_Kd <<" ; "<< this->d_Ki << "]"<< std::endl;

      this->d_Kp = dp[0];
      this->d_Kd = dp[1];
      this->d_Ki = dp[2];
  }
}

/*
void PID::Twiddle(double total_error, double hyperparameter) {
  static double current_best_error = 100000;
  static bool is_twiddle_init = false;
  static bool is_twiddle_reset = false;
  static double last_hyperp = 0;
  std::cout<<"Current best error is: "<< current_best_error<<endl;
  std::cout<<"Dp is: "<<delta_p<<endl;
  if (!is_twiddle_init) {
  	std::cout<<"Twiddle init";
  	current_best_error = total_error;
  	is_twiddle_init = true;
  	return;
  }
  if ((fabs(delta_p) > tolerance)) {
  	if (is_twiddle_reset) {
  		std::cout<<"Twiddle reset!-----------------------------"<<endl;
  		last_hyperp = hyperparameter;
  		hyperparameter += delta_p;
  		std::cout<<"Hyperparameter magnitude increased!"<<endl;
  		is_twiddle_reset = false;
  	} else {
  		if (total_error < current_best_error) {
  			delta_p *= 1.1;
  			is_twiddle_reset = true;
  			current_best_error = total_error;
  		} else {
  			if (fabs(last_hyperp) < fabs(hyperparameter)) {
  				last_hyperp = hyperparameter;
  				hyperparameter -= 2.0 * delta_p;
  				std::cout<<"Hyperparameter magnitude decreased!"<<endl;
  			} else {
  				last_hyperp = hyperparameter;
  				hyperparameter += delta_p;
  				delta_p *= 0.9;
  				std::cout<<"Hyperparameter magnitude kept same!"<<endl;
  				is_twiddle_reset = true;
  			}
  		}
  	}
  }
}
*/