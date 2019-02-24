# Extended Kalman Filter Project Starter Code
Self-Driving Car Engineer Nanodegree Program

## Tasks/ Instructions

In this project you will utilize a kalman filter to estimate the state of a moving object of interest with noisy lidar and radar measurements. Passing the project requires obtaining RMSE values that are lower than the tolerance outlined in the project rubric. 

This project involves the Term 2 Simulator which can be downloaded [here](https://github.com/udacity/self-driving-car-sim/releases)

This repository includes two files that can be used to set up and install [uWebSocketIO](https://github.com/uWebSockets/uWebSockets) for either Linux or Mac systems. For windows you can use either Docker, VMware, or even [Windows 10 Bash on Ubuntu](https://www.howtogeek.com/249966/how-to-install-and-use-the-linux-bash-shell-on-windows-10/) to install uWebSocketIO. Please see the uWebSocketIO Starter Guide page in the classroom within the EKF Project lesson for the required version and installation scripts.

Once the install for uWebSocketIO is complete, the main program can be built and run by doing the following from the project top directory.

1. mkdir build
2. cd build
3. cmake ..
4. make
5. ./ExtendedKF

Tips for setting up your environment can be found in the classroom lesson for this project.

Note that the programs that need to be written to accomplish the project are src/FusionEKF.cpp, src/FusionEKF.h, kalman_filter.cpp, kalman_filter.h, tools.cpp, and tools.h

The program main.cpp has already been filled out, but feel free to modify it.

Here is the main protocol that main.cpp uses for uWebSocketIO in communicating with the simulator.


INPUT: values provided by the simulator to the c++ program

["sensor_measurement"] => the measurement that the simulator observed (either lidar or radar)


OUTPUT: values provided by the c++ program to the simulator

["estimate_x"] <= kalman filter estimated position x
["estimate_y"] <= kalman filter estimated position y
["rmse_x"]
["rmse_y"]
["rmse_vx"]
["rmse_vy"]


[//]: # (Image References)

[image1]: ./examples/EKF_Process.png "Process Measurements"
[image2]: ./examples/Radar_Coordinates.png "Radar Polar Coordinates"
[image3]: ./examples/Kalman_Filter_Update.png "Kalman Filter Update Process"
[image4]: ./examples/Lidar_Measurements.png "Lidar Measurements/State Mapping"
[image5]: ./examples/Radar_H_Function.png "Radar H-Function"
[image6]: ./examples/Radar_H_Linerazation.png "H-Linearzation"
[image7]: ./examples/Radar_H_Jacobian.png "Hj Jacobian Matrix"
[image8]: ./examples/Radar_General_Taylor_Form.png "General Taylor Form"
[image9]: ./examples/EKF_Output.JPG "EKF Output"
[image10]: ./examples/EKF_Output_Analysis.PNG "EKF Output Analysis"

---
## Enviornment

The Project was built locally on Windows 10 - using Docker images 

    #use official docker-image provided by udacity
    `docker pull udacity/controls_kit:latest`

    # Create Container and Mount EKF using -v Flag to real-hardware 
    `docker run -it -p 4567:4567 -v 'pwd':/EKF:/EKF udacity/controls_kit:latest`

## Process Measurements
The code follow the Work-flow of Lidar/Radar Extended kalman-Filter that was detailed described in the lessons.
![alt text][image1]

Although the mathematical proof is rather complex, it turns out that the Kalman Filter equations and EKF equations are very similar, if we consider the following mappings.
![alt text][image3]

1. Lidar:
    
    Has px,py measurements - which can be mapped to the kalman filter states: [px,py,vx,vy] by applying :
        
            H=[1 0 0 0
               0 1 0 0]
    ![alt text][image4]
2. Radar:
    
    For radar it’s more tricky, as Radar uses polar coordinates:
    ![alt text][image5]
    1. h(x') function is used in comparing the measurement to the predicted state.
    ```cpp
        float rho = sqrt(x_(0)*x_(0) + x_(1)*x_(1));
        float phi = atan2(x_(1), x_(0));
        float rho_dot = (x_(0)*x_(2) + x_(1)*x_(3));
    ```

    2. with `y` calculated - phi needed to be normalized to within -pi and pi
    ```cpp
        while (y[1] < -M_PI)
        {
            y[1] += 2 * M_PI;
        } 
        while (y[1] > M_PI)
        {
            y[1] -= 2 * M_PI;
        }
    ```

    2. h(x') needed to be linearized using First Order Taylor Expansion, to maintain Gaussian distibution. 
    ![alt text][image6]
    We can use Taylor-Expansion general formula:
    ![alt text][image8]
    
        and retrieve the jacobian-Matrix
    ![alt text][image7]
    ```cpp
        MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
        MatrixXd Hj(3, 4);
        //recover state parameters
        float px = x_state(0);
        float py = x_state(1);
        float vx = x_state(2);
        float vy = x_state(3);

        //pre-compute a set of terms to avoid repeated calculation
        float c1 = px*px + py*py;
        float c2 = sqrt(c1);
        float c3 = (c1*c2);

        //compute the Jacobian matrix
        Hj << (px / c2), (py / c2), 0, 0,
            -(py / c1), (px / c1), 0, 0,
            py*(vx*py - vy*px) / c3, px*(px*vy - py*vx) / c3, px / c2, py / c2;

        return Hj;
        }
    ```
    
## Results/ Performance

Here is a sample Screen-Shot of Predication vs. Real Measurements from Lidar/Radar.
![alt text][image9]
The recorded RMSE of[X,Y,Vx,Vy] showed the following graph
![alt text][image10]
With a Mean
| Mean(rmse_x)        | Mean(rmse_y)        | Mean(rmse_vx)  | Mean(rmse_vy) |
| ------------- |:-------------:| -----:|-----:|
| 0.097589003      | 0.097589003 | 0.63791807 | 0.513347793 |

