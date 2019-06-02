#!/usr/bin/env bash


pushd ../ros

#rosdep update

find . -name '*.pyc' -delete
catkin_make clean

rm -rf /root/.ros/*
rm -rf ../log/*


catkin_make
source devel/setup.sh
roslaunch launch/styx.launch

cp -R /root/.ros/ ../log

popd