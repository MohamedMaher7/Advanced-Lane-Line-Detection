import sys
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
layers_names = ['yolo_82', 'yolo_94', 'yolo_106']

labels = open(labels_path).read().strip().split("\n")


Frame_Flag = True
iou_threshold = 0.5
Filtered_boxes = []
Filtered_confidence = []
Filtered_classIDs = []


def iou(box1, box2):
    tb = min(box1[0]+0.5*box1[2], box2[0]+0.5*box2[2])-max(box1[0]-0.5*box1[2], box2[0]-0.5*box2[2])
    lr = min(box1[1]+0.5*box1[3], box2[1]+0.5*box2[3])-max(box1[1]-0.5*box1[3], box2[1]-0.5*box2[3])
    if tb < 0 or lr < 0:
        intersection = 0
    else:
        intersection = tb*lr
    return intersection / (box1[2]*box1[3] + box2[2]*box2[3] - intersection)