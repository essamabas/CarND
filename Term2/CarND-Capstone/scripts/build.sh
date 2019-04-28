#!/usr/bin/env bash


pushd ../ros


rm -rf /root/.ros/*
rm -rf ../log/*

# rosdep update
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch

cp -R /root/.ros/ ../log

popd