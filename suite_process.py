import os
import glob
import pickle

# videos = glob.glob('./videos/*.mp4')
videos = ['./videos/clip.mp4']

models = ['facenet', 'ssd-mobilenet-v2']
models_ref = ['face', 'body']

face_feat = [0, 1]

skips = [5, 15, 30]
chips = [0, 1]



for video in videos:
    for model, model_ref in zip(models, models_ref):        
        for skip in skips:
            for chip in chips:
                cmd = 'python3 process_video.py --video={} --network={} --skip={} --chip={} --json=.{}'.format(
                    video, model, skip, chip, 
                    video.split('.')[1]+'_{}_skip{}_chip{}.json'.format(model_ref, skip, chip)
                )
                print('Running '+cmd+'...')
                os.system(cmd)
