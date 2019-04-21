#!/bin/bash

cd ~/catkin_ws/src
git clone https://github.com/udacity/simple_arm_01.git simple_arm

cd ~/catkin_ws
catkin_make
