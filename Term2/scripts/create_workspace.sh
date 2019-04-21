#!/bin/bash

pushd /Term2/Work
mkdir -p catkin_ws/src
cd catkin_ws/src
catkin_init_workspace
cd ..
catkin_make
popd