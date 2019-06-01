#!/usr/bin/env bash


pushd ../ros

# Clean all
rm src/*/*.pyc
rm src/*/*/*.pyc
rm src/*/*/*/*.pyc
catkin_make clean

rm -rf /root/.ros/*
rm -rf ../log/*
# rm -r *.pyc


rosdep update 

# Make 
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch

cp -R /root/.ros/ ../log

popd