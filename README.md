# M2Det_ros

[Original](https://github.com/qijiezhao/M2Det)

[Demo](/gif/demo.gif)


## Preparation
```
$ cd ~
$ cd catkin_ws/src
$ git clone https://github.com/KoutaOhishi/m2det_ros.git && cd m2det_ros
$ sh install.sh
```

### Download weight file(m2det512_vgg.pth)
[Download here](https://drive.google.com/file/d/1NM1UDdZnwHwiNDxhcP-nndaWj24m-90L/view)

```
$ cd ~
$ cp ~/Downloads/m2det512_vgg.pth ~/catkin_ws/src/m2det_ros/weights/
```

## Parameters
- "/m2det_ros/show_result" → show result images or not [defalut : "True"]
- "/m2det_ros/image_topic_name" → name of topic(sensor_msg/Image) to subscribe [default : "/usb_cam/image_raw"]  

If you want to change some parameters, rewrite *m2det_ros.launch* in "/m2det_ros/launch/".
## Run

```
$ roslaunch m2det_ros m2det_ros.launch
```

### Publications:
- /m2det_ros/detect_result [darknet_dnn/BoundingBoxes]
