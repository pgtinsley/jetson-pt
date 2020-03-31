import os
import glob
import pickle

videos = glob.glob('./videos/*.mp4')
models = ['facenet', 'ssd-mobilenet-v2']
skips = [1, 15, 30]
chips = [0, 1]

base = 'python process_frame2cuda.py'

for video in videos:
    for model in models:
        for skip in skips:
            for chip in chips:
                cmd = 'python3 process_frame2cuda.py --video={} --network={} --skip={} --chip={} --pkl=.{} --json=.{}'.format(
                    video, model, skip, chip, video.split('.')[1]+'.pkl', video.split('.')[1]+'.json'
                )
                os.system(cmd)
