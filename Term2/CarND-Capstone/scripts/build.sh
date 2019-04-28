#!/usr/bin/env bash


pushd /Term2/CarND-Capstone/ros


rm -rf /root/.ros/*
rm -rf /Term2/CarND-Capstone/log/*

catkin_make
source devel/setup.sh
roslaunch launch/styx.launch

cp -R /root/.ros/ /Term2/CarND-Capstone/log

popd