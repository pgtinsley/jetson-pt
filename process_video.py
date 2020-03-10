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

import jetson.inference
import jetson.utils

import argparse
import sys

import cv2
import json
import pickle
import face_recognition

# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in a video file using an object detection DNN.",
				formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.detectNet.Usage())

parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use") 
parser.add_argument("--video", type=str, help="path to input video file", required=True)
parser.add_argument("--pkl", type=str, help='path to output Pickle file', required=True)
parser.add_argument("--json", type=str, help='path to output JSON file', required=True)
args = parser.parse_args()

net = jetson.inference.detectNet(args.network, args.threshold)

cap = cv2.VideoCapture(args.video)
if (cap.isOpened()== False): 
    print('Error: could not open video stream.')
    
width = int(cap.get(3))
height = int(cap.get(4))

my_dict = {}

curr_frame = 0
while(cap.isOpened()):

    ret, frame = cap.read()
    
    if ret:
        
        curr_frame += 1
        
        my_dict['frame_'+str(curr_frame)] = {
            'fps': net.GetNetworkFPS(),
        }

        cuda_frame = jetson.utils.cudaFromNumpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA))
        
        detections = net.Detect(cuda_frame, width, height, args.overlay)
        
        detections_dict = {}
        for i, detection in enumerate(detections):
            
            left = int(detection.Left)
            top = int(detection.Top)
            right = int(detection.Right)
            bottom = int(detection.Bottom)            
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                        
            detections_dict['detection_'+str(i)] = {
                'confidence' : detection.Confidence,
                'bbox' : [int(detection.Left), int(detection.Top), int(detection.Right), int(detection.Bottom)],
            }

        my_dict['frame_'+str(curr_frame)]['detections'] = detections_dict

    else:
        break

cap.release()

with open(args.pkl, 'wb') as f:
    pickle.dump(my_dict, f)

my_json = json.dumps(my_dict)

with open(args.json, 'w') as f:
    json.dump(my_json, f)
