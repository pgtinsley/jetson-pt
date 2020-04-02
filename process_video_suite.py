
##### LIBS

import io
import glob
import json
from json import JSONEncoder
import pickle
import os, sys
import argparse

import jetson.utils
import jetson.inference

import cv2
import numpy as np
import face_recognition

from PIL import Image

# network = 'facenet' or 'ssd-mobilenet-v2'

##### FUNCTIONS

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def body_chip0(capture, capture_width, capture_height, net, skip, out_fname):
    
    if not os.path.exists(out_fname):
    
        to_return = {}
        frame_number = 0
    
        while True:
            
            frame_number += 1
    
            if frame_number % skip == 0:
            
                ret, frame = capture.read()
            
                if not ret:
                    break
                
                rgb_frame = frame[:, :, ::-1]
                cuda_frame = jetson.utils.cudaFromNumpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA))
                
                detections = net.Detect(cuda_frame, capture_width, capture_height, 'box,labels,conf')
                
                detections_dict = {}
                detections_dict['fps'] = net.GetNetworkFPS()
                                
                for i, detection in enumerate(detections):
                    bbox = (int(detection.Top), int(detection.Right), int(detection.Bottom), int(detection.Left))
                    detections_dict['det_'+str(i)] = {
                        'conf': detection.Confidence,
                        'bbox': bbox,
                    }
                    
                to_return['frame_'+str(frame_number)] = detections_dict
    
        with open(out_fname, 'w') as f:
            json.dump(to_return, f)

def body_chip1(capture, capture_width, capture_height, net, skip, out_fname):
    
    if not os.path.exists(out_fname):
    
        to_return = {}
        frame_number = 0
    
        while True:
            
            frame_number += 1
    
            if frame_number % skip == 0:
            
                ret, frame = capture.read()
            
                if not ret:
                    break
                
                rgb_frame = frame[:, :, ::-1]
                cuda_frame = jetson.utils.cudaFromNumpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA))
                
                detections = net.Detect(cuda_frame, capture_width, capture_height, 'box,labels,conf')
                
                detections_dict = {}
                detections_dict['fps'] = net.GetNetworkFPS()
                                
                for i, detection in enumerate(detections):
                    bbox = (int(detection.Top), int(detection.Right), int(detection.Bottom), int(detection.Left))
                    chip = Image.fromarray( rgb_frame[bbox[0]: bbox[2], bbox[3]: bbox[1]] )
                    
                    buf = io.BytesIO()
                    chip.save(buf, format='PNG')
                    chip_size = buf.tell()
                    
                    detections_dict['det_'+str(i)] = {
                        'conf': detection.Confidence,
                        'bbox': bbox,
                        'chip_size': chip_size,
                    }
                    
                to_return['frame_'+str(frame_number)] = detections_dict
    
        with open(out_fname, 'w') as f:
            json.dump(to_return, f)
        
def face_feat0_chip0(capture, capture_width, capture_height, net, skip, out_fname):
    
    if not os.path.exists(out_fname):
    
        to_return = {}
        frame_number = 0
    
        while True:
            
            frame_number += 1
    
            if frame_number % skip == 0:
            
                ret, frame = capture.read()
            
                if not ret:
                    break
                
                rgb_frame = frame[:, :, ::-1]
                cuda_frame = jetson.utils.cudaFromNumpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA))
                
                detections = net.Detect(cuda_frame, capture_width, capture_height, 'box,labels,conf')
                
                detections_dict = {}
                detections_dict['fps'] = net.GetNetworkFPS()
                                
                for i, detection in enumerate(detections):
                    bbox = (int(detection.Top), int(detection.Right), int(detection.Bottom), int(detection.Left))
                    detections_dict['det_'+str(i)] = {
                        'conf': detection.Confidence,
                        'bbox': bbox,
                    }
                    
                to_return['frame_'+str(frame_number)] = detections_dict
    
        with open(out_fname, 'w') as f:
            json.dump(to_return, f)
        
def face_feat0_chip1(capture, capture_width, capture_height, net, skip, out_fname):
    
    if not os.path.exists(out_fname):
    
        to_return = {}
        frame_number = 0
    
        while True:
            
            frame_number += 1
    
            if frame_number % skip == 0:
            
                ret, frame = capture.read()
            
                if not ret:
                    break
                
                rgb_frame = frame[:, :, ::-1]
                cuda_frame = jetson.utils.cudaFromNumpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA))
                
                detections = net.Detect(cuda_frame, capture_width, capture_height, 'box,labels,conf')
                
                detections_dict = {}
                detections_dict['fps'] = net.GetNetworkFPS()
                                
                for i, detection in enumerate(detections):
                    bbox = (int(detection.Top), int(detection.Right), int(detection.Bottom), int(detection.Left))
                    chip = Image.fromarray( rgb_frame[bbox[0]: bbox[2], bbox[3]: bbox[1]] )
                    
                    buf = io.BytesIO()
                    chip.save(buf, format='PNG')
                    chip_size = buf.tell()
                    
                    detections_dict['det_'+str(i)] = {
                        'conf': detection.Confidence,
                        'bbox': bbox,
                        'chip_size': chip_size,
                    }
                    
                to_return['frame_'+str(frame_number)] = detections_dict
    
        with open(out_fname, 'w') as f:
            json.dump(to_return, f)
        
def face_feat1_chip0(capture, capture_width, capture_height, net, skip, out_fname):
    
    if not os.path.exists(out_fname):
    
        to_return = {}
        frame_number = 0
    
        while True:
            
            frame_number += 1
    
            if frame_number % skip == 0:
            
                ret, frame = capture.read()
            
                if not ret:
                    break
                
                rgb_frame = frame[:, :, ::-1]
                cuda_frame = jetson.utils.cudaFromNumpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA))
                
                detections = net.Detect(cuda_frame, capture_width, capture_height, 'box,labels,conf')
                
                detections_dict = {}
                detections_dict['fps'] = net.GetNetworkFPS()
                                
                for i, detection in enumerate(detections):
                    bbox = (int(detection.Top), int(detection.Right), int(detection.Bottom), int(detection.Left))
                    enc = face_recognition.face_encodings(rgb_frame, [bbox])[0]
                    detections_dict['det_'+str(i)] = {
                        'conf': detection.Confidence,
                        'bbox': bbox,
                        'enc': enc
                    }
                    
                to_return['frame_'+str(frame_number)] = detections_dict
    
        with open(out_fname, 'w') as f:
            json.dump(to_return, f, cls=NumpyEncoder)
        
def face_feat1_chip1(capture, capture_width, capture_height, net, skip, out_fname):
    
    if not os.path.exists(out_fname):
    
        to_return = {}
        frame_number = 0
    
        while True:
            
            frame_number += 1
    
            if frame_number % skip == 0:
            
                ret, frame = capture.read()
            
                if not ret:
                    break
                
                rgb_frame = frame[:, :, ::-1]
                cuda_frame = jetson.utils.cudaFromNumpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA))
                
                detections = net.Detect(cuda_frame, capture_width, capture_height, 'box,labels,conf')
                
                detections_dict = {}
                detections_dict['fps'] = net.GetNetworkFPS()
                                
                for i, detection in enumerate(detections):
                    bbox = (int(detection.Top), int(detection.Right), int(detection.Bottom), int(detection.Left))
                    chip = Image.fromarray( rgb_frame[bbox[0]: bbox[2], bbox[3]: bbox[1]] )
                    
                    buf = io.BytesIO()
                    chip.save(buf, format='PNG')
                    chip_size = buf.tell()
                    
                    enc = face_recognition.face_encodings(rgb_frame, [bbox])[0]
                    detections_dict['det_'+str(i)] = {
                        'conf': detection.Confidence,
                        'bbox': bbox,
                        'enc': enc,
                        'chip_size': chip_size
                    }
                    
                to_return['frame_'+str(frame_number)] = detections_dict
    
        with open(out_fname, 'w') as f:
            json.dump(to_return, f, cls=NumpyEncoder)
        
#####

def process_video(in_video, network, feat, chip, skip, out_fname):

    capture = cv2.VideoCapture(in_video)
    capture_width  = int(capture.get(3))
    capture_height = int(capture.get(4))

    net = jetson.inference.detectNet(network, 0.5)

    to_return = {}
    frame_number = 0

    if network=='ssd-mobilenet-v2': #body
        if chip==0:
            body_chip0(capture, capture_width, capture_height, net, skip, out_fname)
        else:
            body_chip1(capture, capture_width, capture_height, net, skip, out_fname)
    else: #network=='facenet' # face
        if feat==0:
            if chip==0:
                face_feat0_chip0(capture, capture_width, capture_height, net, skip, out_fname)
            else:
                face_feat0_chip1(capture, capture_width, capture_height, net, skip, out_fname)
        else: #feat==1
            if chip==0:
                face_feat1_chip0(capture, capture_width, capture_height, net, skip, out_fname)
            else:
                face_feat1_chip1(capture, capture_width, capture_height, net, skip, out_fname)

    capture.release()
    
#####

video_fnames = glob.glob('./videos/*.mp4')
for video_fname in video_fnames:
    for network in ['ssd-mobilenet-v2', 'facenet']:
        for feat in [0, 1]:
            for chip in [0, 1]:
                for skip in [5,15,30]:
                    out_fname = video_fname.split('.mp4')[0]+'_'+network+'_feat'+str(feat)+'_chip'+str(chip)+'_skip'+str(skip)+'.json'
                    process_video(video_fname, network, feat, chip, skip, out_fname)
    
    
    
    
    