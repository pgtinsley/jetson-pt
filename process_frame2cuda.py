
##### LIBS

import time
import json
import pickle
import os, sys
import argparse

import jetson.utils
import jetson.inference

import cv2
import numpy as np
import face_recognition

##### ARGS

parser = argparse.ArgumentParser()

parser.add_argument('--network', type=str, default='facenet', help='path to input video file')
parser.add_argument('--overlay', type=str, default='box,labels,conf', help='detection overlay flags')
parser.add_argument('--threshold', type=float, default=0.5, help='minimum detection threshold to use')

parser.add_argument('--video', type=str, help='path to input video file', required=True)
parser.add_argument('--pkl', type=str, help='path to output pickle file', required=True)
parser.add_argument('--json', type=str, help='path to output json file', required=True)

parser.add_argument('--chip', type=int, default=0, help='0 for no chips saved, 1 for saving chips')

args = parser.parse_args()

##### SET UP

net = jetson.inference.detectNet(args.network, args.threshold)

capture = cv2.VideoCapture(args.video)
capture_width  = int(capture.get(3))
capture_height = int(capture.get(4))

to_return = {}
frame_number = 0

while True:
    
    frame_number += 1
    ret, frame = capture.read()

    if not ret:
        break

    rgb_frame = frame[:, :, ::-1]
    cuda_frame = jetson.utils.cudaFromNumpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA))

    detections = net.Detect(cuda_frame, capture_width, capture_height, args.overlay)

    detections_dict = {}
    for i, detection in enumerate(detections):
        bbox = (int(detection.Top), int(detection.Right), int(detection.Bottom), int(detection.Left))
        encs = face_recognition.face_encodings(rgb_frame, [bbox])
        enc = np.zeros(128,)
        if len(encs)>0: enc = encs[0]
        detections_dict['det_'+str(i)] = {
            'bbox': bbox,
            'enc': list(enc),
        }
    
    to_return['frame_'+str(frame_number)] = detections_dict
         
capture.release()

with open(args.pkl, 'wb') as f:
    pickle.dump(to_return, f)

with open(args.json, 'w') as json_file:
  json.dump(to_return, json_file)