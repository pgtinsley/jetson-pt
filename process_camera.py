#!/usr/bin/python
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import jetson.inference
import jetson.utils

import argparse
import ctypes
import sys

import cv2
import json
import pickle

import numpy as np

# parse the command line
parser = argparse.ArgumentParser(description="Segment a live camera stream using an semantic segmentation DNN.", 
						   formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.segNet.Usage())

parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load, see below for options")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use") 

parser.add_argument("--camera", type=str, default="/dev/video0", help="index of the MIPI CSI camera to use (e.g. CSI camera 0)\nor for VL42 cameras, the /dev/video device to use.\nby default, MIPI CSI camera 0 will be used.")
# --camera=/dev/video0
parser.add_argument("--width", type=int, default=1280, help="desired width of camera stream (default is 1280 pixels)")
parser.add_argument("--height", type=int, default=720, help="desired height of camera stream (default is 720 pixels)")

parser.add_argument("--pkl", type=str, default='test.pkl', help='path to output Pickle file')
parser.add_argument("--json", type=str, default='test.json', help='path to output JSON file')
# parser.add_argument("--out", type=str, default='test.avi', help='path to output video file')

args = parser.parse_args()

# load the network
net = jetson.inference.detectNet(args.network, args.threshold)

# create the camera and display
camera = jetson.utils.gstCamera(args.width, args.height, args.camera)

# fps = 10
# out = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc('M','J','P','G'), fps, (args.width, args.height))

my_dict = {}
curr_frame = 0

while curr_frame < 100:

    curr_frame += 1

    my_dict['frame_'+str(curr_frame)] = {
        'fps': net.GetNetworkFPS(),
    }

    # capture the image
    img, width, height = camera.CaptureRGBA(zeroCopy=1)
#     frame = jetson.utils.cudaToNumpy(img, width, height, 4)
#     frame_rgb = np.uint8(cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB))    

    # process the segmentation network
    detections = net.Detect(img, args.width, args.height, args.overlay)
#     print(len(detections))
        
    detections_dict = {}
    for i, detection in enumerate(detections):
        left = int(detection.Left)
        top = int(detection.Top)
        right = int(detection.Right)
        bottom = int(detection.Bottom) 
#         cv2.rectangle(frame_rgb, (left, top), (right, bottom), (0, 0, 255), 2)           
        detections_dict['detection_'+str(i)] = {
            'confidence' : detection.Confidence,
            'bbox' : [int(detection.Left), int(detection.Top), int(detection.Right), int(detection.Bottom)],
        }

#     out.write(frame_rgb)
    
    print(detections_dict)
    my_dict['frame_'+str(curr_frame)]['detections'] = detections_dict

my_dict['frame_1']['fps'] = 0

with open(args.pkl, 'wb') as f:
    pickle.dump(my_dict, f)

my_json = json.dumps(my_dict)

with open(args.json, 'w') as f:
    json.dump(my_json, f)

