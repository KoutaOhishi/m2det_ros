#!/bin/sh

echo "pip install"
pip install torch, torchvision
pip install cython
pip install opencv-python
pip install tqdm

echo "make & catkin_make"
roscd m2det_ros
sh make.sh
cd ../ && cd ../
catkin_make 
