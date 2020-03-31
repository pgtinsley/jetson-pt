import os
import glob
import pickle

videos = glob.glob('./videos/*.mp4')
models = ['facenet', 'ssd-mobilenet-v2']
models_ref = ['face', 'body']
skips = [1, 15, 30]
chips = [0, 1]

base = 'python process_frame2cuda.py'

for video in videos:
    for model, model_ref in zip(models, models_ref):
        for skip in skips:
            for chip in chips:
                cmd = 'python3 process_frame2cuda.py --video={} --network={} --skip={} --chip={} --pkl=.{} --json=.{}'.format(
                    video, model, skip, chip, 
                    video.split('.')[1]+'_{}_skip{}_chip{}.pkl'.format(model_ref, skip, chip), 
                    video.split('.')[1]+'_{}_skip{}_chip{}.json'.format(model_ref, skip, chip)
                )
                os.system(cmd)
