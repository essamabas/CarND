#!/usr/bin/env bash


pushd /CarND-Capstone/ros


rm -rf /root/.ros/*
rm -rf /CarND-Capstone/log/*

catkin_make
source devel/setup.sh
roslaunch launch/styx.launch

cp -R /root/.ros/ /CarND-Capstone/log

popd