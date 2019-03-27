# M2Det_ros

[M2Det](https://github.com/qijiezhao/M2Det):Original


## Preparation
### pip
```
$ pip install torch, torchvision
$ pip install cython
$ pip install opencv-python
$ pip install tqdm
```

### clone repository & catkin_make
```
$ cd ~
$ cd catkin_ws/src
$ git clone https://github.com/KoutaOhishi/m2det_ros.git
$ cd m2det_ros
$ sh make.sh
$ cd ~ && cd catkin_ws/
$ catkin_make
```

### Download weight file
[Download here](https://drive.google.com/file/d/1NM1UDdZnwHwiNDxhcP-nndaWj24m-90L/view)

```
$ cd ~
$ cp ~/Downloads/m2det512_vgg.pth ~/catkin_ws/src/m2det_ros/weights/
```

## Parameters
"/m2det_ros/show_result" : show result images or not  
"/m2det_ros/image_topic_name" : Name of topic(sensor_msg/Image) to subscribe

## Run

```
$ roslaunch m2det_ros m2det_ros.launch
```

### Publications:
- /m2det_ros/detect_result [darknet_dnn/BoundingBoxes]


## Demo
![demo](https://raw.github.com/wiki/KoutaOhishi/m2det_ros/gifs/demo.gif)
