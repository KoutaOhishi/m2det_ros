#! /usr/bin/env python
# -*- coding:utf-8 -*-
import os
import cv2
import numpy as np
import time
import rospy, roslib
from torch.multiprocessing import Pool
from utils.nms_wrapper import nms
from utils.timer import Timer
from configs.CC import Config
import argparse
from layers.functions import Detect, PriorBox
from m2det import build_net
from data import BaseTransform
from utils.core import *
from utils.pycocotools.coco import COCO
from sensor_msgs.msg import Image
from darknet_dnn.msg import *
from cv_bridge import CvBridge, CvBridgeError


#parameters
show_result = rospy.get_param("/m2det_ros/show_result")
subscribe_image = rospy.get_param("/m2det_ros/subscribe_image")

print_info(' -------------------------------------------------------------\n'
           '|                       M2Det ROS                            |\n'
           ' -------------------------------------------------------------', ['blue','bold'])

global cfg
#cfg = Config.fromfile(args.config)
cfg = Config.fromfile(roslib.packages.get_pkg_dir("m2det_ros")+"/configs/m2det512_vgg.py")
anchor_config = anchors(cfg)
print_info('The Anchor info: \n{}'.format(anchor_config))
priorbox = PriorBox(anchor_config)
net = build_net('test',
                size = cfg.model.input_size,
                config = cfg.model.m2det_config)
#init_net(net, cfg, args.trained_model)
init_net(net, cfg, roslib.packages.get_pkg_dir("m2det_ros")+"/weights/m2det512_vgg.pth")

print_info('===> Finished constructing and loading model',['yellow','bold'])
net.eval()
with torch.no_grad():
    priors = priorbox.forward()
    if cfg.test_cfg.cuda:
        net = net.cuda()
        priors = priors.cuda()
        #cudnn.benchmark = True
    else:
        net = net.cpu()
_preprocess = BaseTransform(cfg.model.input_size, cfg.model.rgb_means, (2, 0, 1))
detector = Detect(cfg.model.m2det_config.num_classes, cfg.loss.bkg_label, anchor_config)
cats = [_.strip().split(',')[-1] for _ in open(roslib.packages.get_pkg_dir("m2det_ros")+'/data/coco_labels.txt','r').readlines()]
labels = tuple(['__background__'] + cats)

print_info('===> Detection Start',['blue','bold'])

def _to_color(indx, base):
    """ return (b, r, g) tuple"""
    base2 = base * base
    b = 2 - indx / base2
    r = 2 - (indx % base2) / base
    g = 2 - (indx % base2) % base
    return b * 127, r * 127, g * 127
base = int(np.ceil(pow(cfg.model.m2det_config.num_classes, 1. / 3)))
colors = [_to_color(x, base) for x in range(cfg.model.m2det_config.num_classes)]


def pub_result(im, bboxes, scores, cls_inds, thr=0.2):
    imgcv = np.copy(im)
    h, w, _ = imgcv.shape
    boundingBoxes = BoundingBoxes()

    for i, box in enumerate(bboxes):
        if scores[i] < thr:
            continue
        cls_indx = int(cls_inds[i])
        box = [int(_) for _ in box]
        mess = '%s: %.3f' % (labels[cls_indx], scores[i])

        boundingBox = BoundingBox()
        boundingBox.Class = labels[cls_indx]
        boundingBox.probability = scores[i]
        boundingBox.x = box[0]
        boundingBox.y = box[1]
        boundingBox.width = box[2]
        boundingBox.height = box[3]
        boundingBoxes.boundingBoxes.append(boundingBox)

        if show_result == True:
            thick = int((h + w) / 300)
            cv2.rectangle(imgcv,
                          (box[0], box[1]), (box[2], box[3]),
                          colors[cls_indx], thick)
            cv2.putText(imgcv, mess, (box[0], box[1] - 7),
                        0, 1e-3 * h, colors[cls_indx], thick // 3)

    pub_bbox = rospy.Publisher("m2det_ros/detetct_result", BoundingBoxes, queue_size=1)
    pub_bbox.publish(boundingBoxes)

    if show_result == True:
        cv2.imshow("result", imgcv)
        cv2.waitKey(1)

captureImage = Image()

def imgCallback(msg):
    global captureImage
    captureImage = msg

    try:
        cv_img = CvBridge().imgmsg_to_cv2(captureImage, "bgr8")

        w,h =cv_img.shape[1],cv_img.shape[0]
        img = _preprocess(cv_img).unsqueeze(0)

        if cfg.test_cfg.cuda:
            img = img.cuda()

        scale = torch.Tensor([w,h,w,h])
        out = net(img)

        boxes, scores = detector.forward(out, priors)
        boxes = (boxes[0]*scale).cpu().numpy()
        scores = scores[0].cpu().numpy()
        allboxes = []
        for j in range(1, cfg.model.m2det_config.num_classes):
            inds = np.where(scores[:,j] > cfg.test_cfg.score_threshold)[0]
            if len(inds) == 0:
                continue
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(np.float32, copy=False)
            soft_nms = cfg.test_cfg.soft_nms
            keep = nms(c_dets, cfg.test_cfg.iou, force_cpu = soft_nms) #min_thresh, device_id=0 if cfg.test_cfg.cuda else None)
            keep = keep[:cfg.test_cfg.keep_per_class]
            c_dets = c_dets[keep, :]
            allboxes.extend([_.tolist()+[j] for _ in c_dets])

        allboxes = np.array(allboxes)
        boxes = allboxes[:,:4]
        scores = allboxes[:,4]
        cls_inds = allboxes[:,5]

        pub_result(cv_img, boxes, scores, cls_inds)

    except CvBridgeError as e:
         print(e)



if __name__ == "__main__":
    rospy.init_node("m2det_ros")
    rospy.Subscriber(subscribe_image, Image, imgCallback)

    rospy.spin()
