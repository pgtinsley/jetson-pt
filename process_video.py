
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

from PIL import Image

##### ARGS

parser = argparse.ArgumentParser()

parser.add_argument('--network', type=str, default='facenet', help='path to input video file')
parser.add_argument('--overlay', type=str, default='box,labels,conf', help='detection overlay flags')
parser.add_argument('--threshold', type=float, default=0.5, help='minimum detection threshold to use')

parser.add_argument('--video', type=str, help='path to input video file', required=True)
parser.add_argument('--json', type=str, help='path to output json file', required=True)

parser.add_argument('--skip', type=int, default=15, help='integer interval to process every Xth frame')
parser.add_argument('--chip', type=int, default=0, help='0 = do NOT save chips; 1 = save chips')

args = parser.parse_args()

##### SET UP

net = jetson.inference.detectNet(args.network, args.threshold)

capture = cv2.VideoCapture(args.video)
capture_width  = int(capture.get(3))
capture_height = int(capture.get(4))

to_return = {}
frame_number = 0

if args.network=='facenet':

    while True:
        
        frame_number += 1
        
        if  frame_number % args.skip == 0:
        
            ret, frame = capture.read()
        
            if not ret:
                break
        
            rgb_frame = frame[:, :, ::-1]
            cuda_frame = jetson.utils.cudaFromNumpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA))
            
            detections = net.Detect(cuda_frame, capture_width, capture_height, args.overlay)
            
            detections_dict = {}
            detections_dict['fps'] = net.GetNetworkFPS()
            
            if args.chip == 0: # do NOT save chips     
            
                for i, detection in enumerate(detections):
                    bbox = (int(detection.Top), int(detection.Right), int(detection.Bottom), int(detection.Left))
                    enc = face_recognition.face_encodings(rgb_frame, [bbox])[0]
                    detections_dict['det_'+str(i)] = {
                        'bbox': bbox,
                        'enc': list(enc),
                    }
            
            else: # do save chips
            
                for i, detection in enumerate(detections):
                    bbox = (int(detection.Top), int(detection.Right), int(detection.Bottom), int(detection.Left))
                    enc = face_recognition.face_encodings(rgb_frame, [bbox])[0]
                    chip = Image.fromarray( rgb_frame[bbox[0]: bbox[2], bbox[3]: bbox[1]] )
                    detections_dict['det_'+str(i)] = {
                        'bbox': bbox,
                        'enc': list(enc),
                        'chip': chip,
                    }
            
            to_return['frame_'+str(frame_number)] = detections_dict

else:

    while True:
        
        frame_number += 1
        
        if  frame_number % args.skip == 0:
        
            ret, frame = capture.read()
        
            if not ret:
                break
        
            rgb_frame = frame[:, :, ::-1]
            cuda_frame = jetson.utils.cudaFromNumpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA))
        
            detections = net.Detect(cuda_frame, capture_width, capture_height, args.overlay)
        
            detections_dict = {}
            detections_dict['fps'] = net.GetNetworkFPS()

            if args.chip == 0: # do NOT save chips     
            
                for i, detection in enumerate(detections):
                    bbox = (int(detection.Top), int(detection.Right), int(detection.Bottom), int(detection.Left))
                    detections_dict['det_'+str(i)] = {
                        'bbox': bbox,
                        'enc': list(enc),
                    }
            
            else: # do save chips
            
                for i, detection in enumerate(detections):
                    bbox = (int(detection.Top), int(detection.Right), int(detection.Bottom), int(detection.Left))
                    enc = face_recognition.face_encodings(rgb_frame, [bbox])[0]
                    chip = Image.fromarray( rgb_frame[bbox[0]: bbox[2], bbox[3]: bbox[1]] )
                    detections_dict['det_'+str(i)] = {
                        'bbox': bbox,
                        'enc': list(enc),
                        'chip': chip,
                    }
            
            to_return['frame_'+str(frame_number)] = detections_dict
         
capture.release()

with open(args.json, 'w') as json_f:
    json.dump(to_return, json_f)